
import os
import streamlit as st
from dotenv import load_dotenv
from connect_faiss_langchain import load_vectorstore
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load env variables
load_dotenv()

st.set_page_config(page_title="Deep Learning Research Assistant")

# Get chain (with error handling)

@st.cache_resource
def get_chain():
    try:
        vectorstore = load_vectorstore()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(" OPENAI_API_KEY is missing from .env")

        llm = ChatOpenAI(openai_api_key=api_key)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )

        return qa_chain

    except Exception as e:
        st.error(f"Failed to load RetrievalQA chain: {e}")
        return None

qa_chain = get_chain()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.title(" Deep Learning Research Assistant")

if qa_chain:
    col1, col2 = st.columns([4, 1])
    query = col1.text_input("Your question:")
    ask = col2.button("Ask")
    clear = st.button(" Clear Chat")

    if ask and query:
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"query": query})
                st.session_state.chat_history.append({
                    "question": query,
                    "answer": result["result"]
                })
            except Exception as e:
                st.error(f"Error during answering: {e}")

    if clear:
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

    # Display chat
    if st.session_state.chat_history:
        st.markdown("Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**Q{i+1}:** {chat['question']}")
            st.markdown(f"**A{i+1}:** {chat['answer']}")
            st.markdown("---")

else:
    st.warning("RetrievalQA chain not loaded — check your model, vectorstore, or OpenAI key.")

