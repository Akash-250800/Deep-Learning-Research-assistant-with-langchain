import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def load_vectorstore():
    if not os.path.exists("faiss_index.bin") or not os.path.exists("documents.pkl"):
        raise FileNotFoundError("Missing documents.pkl or faiss_index.bin")

    embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.load_local(".", embedding_model, allow_dangerous_deserialization=True)

