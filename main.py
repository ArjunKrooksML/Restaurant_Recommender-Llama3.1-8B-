# main.py
import streamlit as st
import pandas as pd
import config
from dataprocess import loadcsv, process_data
from vector_store import embedding_model, vector_store as build_vector_store_func
from rag_pipe import get_llm, rag_seq, get_answer_from_chain
import traceback

st.set_page_config(page_title="Restaurant Q&A Chatbot", layout="wide")

@st.cache_resource
def embed_cache():
    return embedding_model()

@st.cache_resource
def llm_cache():
    return get_llm()

@st.cache_resource(show_spinner="Loading Vector DB...")
def cached_vs(_data_path):
    try:
        df = loadcsv(_data_path)
        if df is None:
            st.error(f"Failed to load data from '{_data_path}'.")
            return None
        docs = process_data(df)
        if not docs:
            st.error("No documents were processed. Check CSV or processing logic.")
            return None
        embeddings = embed_cache()
        if embeddings is None:
            st.error("Failed to load embedding model.")
            return None
        vs = build_vector_store_func(docs=docs, embeddings=embeddings, db_p=config.DB_PATH, create_new=True)
        if vs is None:
            st.error("Failed to create or load vector store.")
            return None
        return vs
    except FileNotFoundError:
        st.error(f"The data file '{_data_path}' was not found.")
        return None
    except Exception as e:
        st.error(f"Error during data processing or vector store creation: {e}")
        traceback.print_exc()
        return None

st.title("🍽️ Zomato Restaurant Q&A Chatbot")
st.caption(f"Powered by LangChain & Ollama ({config.OLLAMA_MODEL})")

llm_instance = None
vector_store = None
try:
    llm_instance = llm_cache()
    vector_store = cached_vs(config.DATA_FILE)
except Exception as load_error:
    st.error(f"Fatal error loading components: {load_error}")
    traceback.print_exc()
    st.stop()

if vector_store and llm_instance:
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": config.TOP_K})
        rag_chain_instance = rag_seq(retriever, llm_instance)
    except Exception as chain_error:
        st.error(f"Failed to create RAG chain: {chain_error}")
        traceback.print_exc()
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything about the restaurants in the dataset."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask about restaurants...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = get_answer_from_chain(user_query, rag_chain_instance)
                    answer = response.get("answer", "Sorry, I couldn't generate an answer.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error processing your query: {e}")
                    traceback.print_exc()
                    st.session_state.messages.append({"role": "assistant", "content": "Sorry, an error occurred while processing your query."})
else:
    st.warning("Knowledge base or language model could not be loaded.")
