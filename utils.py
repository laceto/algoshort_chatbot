# Function to embed and add long text data in chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


import pandas as pd
import numpy as np
import ast

from typing import Optional
from langchain_core.runnables import Runnable, RunnableConfig
from langchain.schema import Document
from openai import OpenAI  # or your OpenAI client import

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def create_BM25retriever_from_docs(
    docs: list[Document], 
    k : int
    ):  
    """  
    Create a BM25 retriever from the provided documents.  
  
    Args:  
        docs (list[Document]): List of LangChain Document objects.
        k (int): Number of top documents to retrieve.  
  
    Returns:  
        BM25Retriever: BM25 retriever instance configured with the provided documents and k value.  
  
    Raises:  
        ValueError: If the documents list is empty or if k is not a positive integer.  
        Exception: For any other error that may occur during the BM25 retriever creation.  
    """  
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
    """  
    Create a hybrid retriever that combines a sparse retriever and a semantic retriever.  
  
    Args:  
        sparse_retriever: Instance of a sparse retriever.  
        semantic_retriever: Instance of a semantic retriever.  
        weights_sparse (float): The weight to assign to the sparse retriever,   
                                which should be between 0 and 1.  
  
    Returns:  
        EnsembleRetriever: Ensemble retriever that combines the two retrievers.  
  
    Raises:  
        ValueError: If the weights_sparse is not between 0 and 1.  
        Exception: For any other error that may occur during the hybrid retriever creation.  
    """  
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


from openai import OpenAI
def get_query_embeddings(query, embeddings_model_name):
    client = OpenAI()

    response = client.embeddings.create(
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

def get_embeddings_from_csv(path_to_csv_embedding = './book_embeddings.csv', embedding_column = 'embedding'):
    df_embeddings = pd.read_csv('./book_embeddings.csv')
    # # Convert the text column to numpy arrays
    df_embeddings["embedding_array"] = df_embeddings[embedding_column].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))

    embeddings = df_embeddings['embedding_array']
    embeddings = np.stack(embeddings)
    return embeddings

def get_embedding_dim(embeddings):
    embedding_dims = embeddings.shape
    embedding_dim = embedding_dims[1]
    return embedding_dim

def create_vectorstore(docs, embeddings, fake_embeddings_model):
    # docs = load_book('./data/book.txt')

    # Assume docs is your list of Document objects
    # for i, doc in enumerate(docs, start=1):
    #     # Copy existing metadata or create new dict if None
    #     metadata = dict(doc.metadata) if doc.metadata else {}
    #     # Add or overwrite the 'id' field with progressive integer
    #     metadata["id"] = i
    #     # Update the document's metadata
    #     doc.metadata = metadata

    
    # df_embeddings = pd.read_csv('./book_embeddings.csv')
    # # # Convert the text column to numpy arrays
    # df_embeddings["embedding_array"] = df_embeddings['embedding'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))

    # embeddings = df_embeddings['embedding_array']
    # embeddings = np.stack(embeddings)
    # embedding_dims = embeddings.shape

    # # Create a FAISS index for the embedding dimension
    embedding_dim = get_embedding_dim(embeddings)
    # print(embedding_dim)
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index

    # Add embeddings to the index
    index.add(embeddings)

    # Create an in-memory docstore mapping from internal index to document ID
    index_to_docstore_id = {i: doc.metadata["id"] for i, doc in enumerate(docs)}

    # Create the docstore with documents keyed by their IDs
    docstore = InMemoryDocstore({doc.metadata["id"]: doc for doc in docs})

    # Create a fake embeddings object with the correct embedding dimension
    # fake_embeddings = FakeEmbeddings(size=embedding_dim)

    # Initialize the FAISS vector store
    vector_store = FAISS(
        embedding_function=fake_embeddings_model,  # Not used since embeddings are precomputed
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    # path_vectorstore = "./vectorstore/faiss_index"
    # vector_store.save_local(path_vectorstore, index_name='faiss_index')

    # Now you can use the vector store as a retriever
    # retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    return vector_store

# Define the state schema
class State(TypedDict):
    question: str
    context: str
    answer: str

# def embed_and_add_text(vector_store, docs: Document):
#     vector_store.add_documents(docs)
#     # Persist to disk
#     vector_store._collection._client.persist()
#     # return vector_store

# def retrieve(retriever, state: State) -> State:
#     docs = retriever.get_relevant_documents(state["question"])
#     context = "\n\n".join(doc.page_content for doc in docs)
#     return {**state, "context": context}

# def generate(llm, state: State) -> State:
#     messages = [
#         SystemMessage(content="You are a helpful assistant. Use the provided context to answer the question. If you don't know the answer, say so."),
#         HumanMessage(content=f"Context: {state['context']}\n\nQuestion: {state['question']}"),
#     ]
#     response = llm.invoke(messages)
#     return {**state, "answer": response.content}


