import streamlit as st
from dotenv import load_dotenv
import os
from connect_faiss_langchain import load_vectorstore
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Deep Learning Research Assistant")
st.title(" Deep Learning Research Assistant")

@st.cache_resource
def get_chain():
    try:
        vectorstore = load_vectorstore()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing.")
        llm = ChatOpenAI(openai_api_key=api_key)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    except Exception as e:
        st.error(f"Error loading chain: {e}")
        return None

qa_chain = get_chain()

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a research question:")

if st.button("Ask") and query:
    if qa_chain:
        result = qa_chain.invoke({"query": query})
        st.session_state.chat_history.append((query, result["result"]))
    else:
        st.error("Retrieval chain not available.")

if st.button("Clear"):
    st.session_state.chat_history = []

# Display chat history
for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
    st.markdown("---")

