from utils import get_embeddings_from_csv, load_book, get_embedding_dim, RetrieverRunnable, create_BM25retriever_from_docs, create_hybrid_retriever
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

# import numpy as np

# fake_embeddings_model = FakeEmbeddings(size=get_embedding_dim(get_embeddings_from_csv()))
# vector_store = FAISS.load_local(path_vectorstore, index_name='faiss_index_book', embeddings=fake_embeddings_model, allow_dangerous_deserialization=True)

# from langchain import hub
# from langchain_core.output_parsers import StrOutputParser
# prompt = hub.pull("rlm/rag-prompt")
# rag_chain = prompt | llm | StrOutputParser()

# k_docs = 10

# config = {
#     "configurable": {
#         "embeddings_model_name": "text-embedding-3-large",
#         "k": k_docs,
#     }
# }

# query = 'describe who were the turtle traders, what kind of strategies the had, who were the founders, what they traded and their results'

# def get_hybrid_retrieve(docs, k_docs, vector_store, embeddings_model_name, weights_sparse):
#     bm25_retriever = create_BM25retriever_from_docs(docs, k=k_docs)
#     retriever = RetrieverRunnable(vector_store=vector_store, default_embedding_model=embeddings_model_name)
#     hybrid_retriever = create_hybrid_retriever(bm25_retriever, retriever, weights_sparse=weights_sparse)
#     return hybrid_retriever

# def retrieve(retriever, question):
#     docs = retriever.invoke(question)
#     context = "\n\n".join(doc.page_content for doc in docs)
#     return context

# retriever = get_hybrid_retrieve(load_book('./data/book.txt'), k_docs, vector_store, embeddings_model_name, weights_sparse=1)
# context = retrieve(retriever, query)


# from langchain_core.messages import HumanMessage, SystemMessage
# def generate(llm, context, question):
#     messages = [
#         SystemMessage(content="You are a helpful assistant. Use the provided context to answer the question. If you don't know the answer, say so."),
#         HumanMessage(content=f"Context: {context}\n\nQuestion: {question}"),
#     ]
#     llm = llm | StrOutputParser()
#     response = llm.invoke(messages)
#     return response



# rich.print(
#     generate(llm=llm, context=context, question=query)
# )





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
