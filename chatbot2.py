from typing import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI
from langchain_core.runnables import Runnable
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import FakeEmbeddings

from dotenv import load_dotenv 
import pandas as pd
import numpy as np
import ast
from typing import Optional

def create_BM25retriever_from_docs(
    docs: list[Document], 
    k : int
    ):  

    try:  
        if not docs:  
            raise ValueError("The documents list cannot be empty.")  
        if k <= 0:  
            raise ValueError("k must be a positive integer.")  
  
        bm25_retriever = BM25Retriever.from_documents(docs)  
        bm25_retriever.k = k  
        return bm25_retriever  
    except Exception as e:  
        print(f"An error occurred while creating the BM25 retriever: {e}")  
        return None  
    
def create_hybrid_retriever(
    sparse_retriever, 
    semantic_retriever,
     weights_sparse : float
     ):  

    try:  
        if not (0 <= weights_sparse <= 1):  
            raise ValueError("weights_sparse must be between 0 and 1.")  
  
        ensemble_retriever = EnsembleRetriever(  
            retrievers=[sparse_retriever, semantic_retriever],  
            weights=[weights_sparse, 1 - weights_sparse]  
        )  
        return ensemble_retriever  
    except Exception as e:  
        print(f"An error occurred while creating the hybrid retriever: {e}")  
        return None  

class RetrieverRunnable(Runnable):
    def __init__(self, vector_store, default_embedding_model: str = "text-embedding-3-large"):
        self.vector_store = vector_store
        self.default_embedding_model = default_embedding_model
        self.client = OpenAI()  # instantiate once

    def invoke(self, input: str, config: Optional[dict] = None) -> str:
        embeddings_model_name = self.default_embedding_model
        k = 10
        if config and "configurable" in config:
            embeddings_model_name = config["configurable"].get("embeddings_model_name", embeddings_model_name)
            k = config["configurable"].get("k", k)
        context = self.retrieve_and_format(input, embeddings_model_name, k)
        return context
    
    def format_docs(self, docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    
    def retrieve_and_format(self, query: str, embeddings_model_name: str, k: int) -> str:
        # Embed the query externally (your embedding function)
        query_embedding = self.get_query_embeddings(query, embeddings_model_name)

        # Retrieve docs using vector store similarity search by vector
        docs = self.vector_store.similarity_search_by_vector(query_embedding, k=k)

        # Extract metadata from docs
        metadata_list = [doc.metadata for doc in docs]

        # Format docs into context string
        # context = self.format_docs(docs)

        return docs
        
    def get_query_embeddings(self, query: str, embeddings_model_name: str):
        response = self.client.embeddings.create(
            input=query,
            model=embeddings_model_name,
            dimensions=1024
        )
        return response.data[0].embedding

def load_book(path_data, chunk_size: int = 500, chunk_overlap: int = 100):
    text_loader_kwargs = {"autodetect_encoding": True}
    loader = TextLoader(path_data, **text_loader_kwargs)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    clean_docs = splitter.split_documents(docs)
    for doc in clean_docs:
        # Remove all newline characters
        doc.page_content = doc.page_content.replace("\n", " ").strip()
        doc.page_content = doc.page_content.replace("\t", " ").strip()

    for i, doc in enumerate(clean_docs, start=1):
        # Copy existing metadata or create new dict if None
        metadata = dict(doc.metadata) if doc.metadata else {}
        # Add or overwrite the 'id' field with progressive integer
        metadata["id"] = i
        # Update the document's metadata
        doc.metadata = metadata

    return clean_docs
    
def get_hybrid_retrieve(docs, k_docs, vector_store, embeddings_model_name, weights_sparse):
    bm25_retriever = create_BM25retriever_from_docs(docs, k=k_docs)
    retriever = RetrieverRunnable(vector_store=vector_store, default_embedding_model=embeddings_model_name)
    hybrid_retriever = create_hybrid_retriever(bm25_retriever, retriever, weights_sparse=weights_sparse)
    return hybrid_retriever

def get_embedding_dim(embeddings):
    embedding_dims = embeddings.shape
    embedding_dim = embedding_dims[1]
    return embedding_dim

def get_embeddings_from_csv(path_to_csv_embedding = './book_embeddings.csv', embedding_column = 'embedding'):
    df_embeddings = pd.read_csv('./book_embeddings.csv')
    # # Convert the text column to numpy arrays
    df_embeddings["embedding_array"] = df_embeddings[embedding_column].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))

    embeddings = df_embeddings['embedding_array']
    embeddings = np.stack(embeddings)
    return embeddings

class State(TypedDict):
    question: str
    context: str
    answer: str

def make_generate_node(llm):
    def generate(state: State) -> State:
        messages = [
            SystemMessage(content="You are a helpful assistant. Use the provided context to answer the question. If you don't know the answer, say so."),
            HumanMessage(content=f"Context: {state['context']}\n\nQuestion: {state['question']}"),
        ]
        llm_with_parser = llm | StrOutputParser()
        response = llm_with_parser.invoke(messages)
        return {**state, "answer": response}
    return generate

def retrieve(state: State) -> State:
    docs = hybrid_retriever.invoke(state["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    return {**state, "context": context}

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
embeddings_model_name = "text-embedding-3-small"
embeddings_model = OpenAIEmbeddings(model=embeddings_model_name)
path_vectorstore = "./vectorstore/book"
fake_embeddings_model = FakeEmbeddings(size=get_embedding_dim(get_embeddings_from_csv()))
vector_store = FAISS.load_local(path_vectorstore, index_name='faiss_index_book', embeddings=fake_embeddings_model, allow_dangerous_deserialization=True)


docs = load_book('./data/book.txt')
bm25_retriever = create_BM25retriever_from_docs(docs, k=10)
retriever_runnable = RetrieverRunnable(vector_store=vector_store, default_embedding_model=embeddings_model_name)
hybrid_retriever = create_hybrid_retriever(bm25_retriever, retriever_runnable, weights_sparse=0.5)



generate = make_generate_node(llm)

graph = (
    StateGraph(State)
    .add_node("retrieve", retrieve)
    .add_node("generate", generate)
    .add_edge(START, "retrieve")
    .add_edge("retrieve", "generate")
    .add_edge("generate", END)
    .compile()
)

st.title("LangGraph RAG Chatbot with Chroma")

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask a question:")

if question:
    with st.spinner("Thinking..."):
        result = graph.invoke({"question": question, "context": "", "answer": ""})
        answer = result["answer"]
        st.session_state.chat_history.append(("User", question))
        st.session_state.chat_history.append(("Bot", answer))

for speaker, message in st.session_state.chat_history:
    if speaker == "User":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
