from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pickle, os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY missing")

docs = [
    Document(page_content="Deep learning is powerful."),
    Document(page_content="Transformers dominate NLP."),
    Document(page_content="FAISS handles vector search.")
]

with open("documents.pkl", "wb") as f:
    pickle.dump(docs, f)

embedding = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.from_documents(docs, embedding)
vectorstore.save_local(".")

