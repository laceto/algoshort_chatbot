from utils import get_embeddings_from_csv, create_vectorstore, load_book, get_embedding_dim, get_query_embeddings, RetrieverRunnable
from dotenv import load_dotenv 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
import rich

set_llm_cache(SQLiteCache(database_path=".langchain.db"))
load_dotenv()

# Initialize LLM and embeddings
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
embeddings_model_name = "text-embedding-3-small"
embeddings_model = OpenAIEmbeddings(model=embeddings_model_name)
path_vectorstore = "./vectorstore/book"

######################### create vs and save it for later use
# # load text to embed, load embeddings got from openai, create vs and save it
# docs = load_book('./data/book.txt')
# embeddings = get_embeddings_from_csv()
# fake_embeddings_model = FakeEmbeddings(size=get_embedding_dim(embeddings))
# vector_store = create_vectorstore(docs, embeddings, fake_embeddings_model)
# vector_store.save_local(path_vectorstore, index_name='faiss_index_book')
######################### create vs and save it for later use

import numpy as np

fake_embeddings_model = FakeEmbeddings(size=get_embedding_dim(get_embeddings_from_csv()))
vector_store = FAISS.load_local(path_vectorstore, index_name='faiss_index_book', embeddings=fake_embeddings_model, allow_dangerous_deserialization=True)

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
prompt = hub.pull("rlm/rag-prompt")
rag_chain = prompt | llm | StrOutputParser()

config = {
    "configurable": {
        "embeddings_model_name": "text-embedding-3-large",
        "k": 5,
    }
}
query = 'what is the kelly criterion'
retriever = RetrieverRunnable(vector_store=vector_store, default_embedding_model=embeddings_model_name)

print(rag_chain.invoke({
    'question': query,
    'context': retriever.invoke(query, config)
}))




# print(vector_store.index.d)

# query = "why to use relative prices"
# query_embedding = get_query_embeddings(query, embeddings_model_name)

# query_embedding=np.array(query_embedding, dtype=np.float32)
# print(len(query_embedding))

# Perform similarity search by vector
# results = vector_store.similarity_search_by_vector(query_embedding, k=5)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# def retrieve_and_format(query: str, embeddings_model_name:str, k:int):
#     # Embed the query externally (your embedding function)
#     query_embedding = get_query_embeddings(query, embeddings_model_name)  # your embedding function returning np.array

#     # Retrieve docs using vector store similarity search by vector
#     docs = vector_store.similarity_search_by_vector(query_embedding, k=k)

#     # Format docs into context string
#     context = format_docs(docs)
#     return context

# # Define a custom Runnable to integrate retrieval step
# class RetrieverRunnable:
#     def invoke(self, query: str, embeddings_model_name:str, k:int) -> str:
#         return retrieve_and_format(query, embeddings_model_name, k)

# from langchain import hub
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# # docs_retr = retrieve_and_format(query=query, embeddings_model_name=embeddings_model_name, k= 10)
# retriever = RetrieverRunnable()
# docs_retr = retriever.invoke(query=query, embeddings_model_name=embeddings_model_name, k= 10)

# prompt = hub.pull("rlm/rag-prompt")

# rich.print(prompt)

# from rich import print as rich_print
# # Build the RAG chain
# rag_chain = (
#     {"context": RetrieverRunnable(), "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # Run the chain with your query
# query = "describe the regime detection using floor ceiling method"
# result = rag_chain.invoke(query, embeddings_model_name, 15)
# rich.print(docs_retr)


# from langchain_core.documents import Document

# import streamlit as st
# from langgraph.graph import StateGraph, START, END

# print(embeddings)

# import pandas as pd
# import numpy as np
# import ast

# docs = load_book('./data/book.txt')

# # Assume docs is your list of Document objects
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
# embedding_dim = embedding_dims[1]
# # print(embedding_dim)
# index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index

# # Add embeddings to the index
# index.add(embeddings)

# # Create an in-memory docstore mapping from internal index to document ID
# index_to_docstore_id = {i: doc.metadata["id"] for i, doc in enumerate(docs)}

# # Create the docstore with documents keyed by their IDs
# docstore = InMemoryDocstore({doc.metadata["id"]: doc for doc in docs})

# # Create a fake embeddings object with the correct embedding dimension
# fake_embeddings = FakeEmbeddings(size=embedding_dim)

# # Initialize the FAISS vector store
# vector_store = FAISS(
#     embedding_function=fake_embeddings,  # Not used since embeddings are precomputed
#     index=index,
#     docstore=docstore,
#     index_to_docstore_id=index_to_docstore_id,
# )

# # Now you can use the vector store as a retriever
# retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# vectorstore = create_vectorstore()
# path_vectorstore = "./vectorstore/faiss_index"
# from langchain_community.vectorstores import FAISS
# db = FAISS.load_local(path_vectorstore, embeddings, allow_dangerous_deserialization=True)

# query = "describe the regime detection using floor ceiling method"
# query_embedding = embeddings_model.embed_query(query)

# # Perform similarity search by vector
# retr_docs = vectorstore.similarity_search_by_vector(query_embedding, k=10)
# rich.print(retr_docs)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# prompt = hub.pull("rlm/rag-prompt")

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# rich.print(rag_chain.invoke(query))

# path_vectorstore = "./vectorstore/faiss_index"

# def embed_docs(docs, embeddings, path_vectorstore):
#     db = FAISS.from_documents(docs[:1], embeddings)
#     batch_size = 5
#     for i in range(1, len(docs), batch_size):
#         batch = docs[i : i + batch_size]
#         db.add_documents(batch)
#     db.save_local(path_vectorstore)

# embed_docs(docs, embeddings, path_vectorstore)


# db = FAISS.load_local(path_vectorstore, embeddings, allow_dangerous_deserialization=True)

# query = "What this book covers at chapter 1?"

# retriever = db.as_retriever()

# prompt = hub.pull("rlm/rag-prompt")

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# rich.print(rag_chain.invoke(query))



# graph = (
#     StateGraph(State)
#     .add_node("retrieve", retrieve)
#     .add_node("generate", generate)
#     .add_edge(START, "retrieve")
#     .add_edge("retrieve", "generate")
#     .add_edge("generate", END)
#     .compile()
# )

# st.title("LangGraph RAG Chatbot with Chroma")

# # Text input for embedding new data
# with st.expander("Embed new data"):
#     new_text = st.text_area("Paste your long text here to embed and add to the vector store:")
#     if st.button("Embed and Add"):
#         if new_text.strip():
#             with st.spinner("Embedding and adding documents..."):
#                 embed_and_add_text(new_text)
#             st.success("Text embedded and added to vector store!")

# # Chat interface
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# question = st.text_input("Ask a question:")

# if question:
#     with st.spinner("Thinking..."):
#         result = graph.invoke({"question": question, "context": "", "answer": ""})
#         answer = result["answer"]
#         st.session_state.chat_history.append(("User", question))
#         st.session_state.chat_history.append(("Bot", answer))

# for speaker, message in st.session_state.chat_history:
#     if speaker == "User":
#         st.markdown(f"**You:** {message}")
#     else:
#         st.markdown(f"**Bot:** {message}")
