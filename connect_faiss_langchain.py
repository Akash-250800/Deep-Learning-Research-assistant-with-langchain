import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document

def load_vectorstore():

    # 1. Load Documents and FAISS Index

    print("Loading documents...")
    with open('documents.pkl', 'rb') as f:
        documents = pickle.load(f)

    print(f"Loaded {len(documents)} documents.")

    print("Loading FAISS index...")
    index = faiss.read_index('faiss_index.bin')


    # 2. Build Mappings


    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    # VERY IMPORTANT: Wrap each document properly
    docstore = InMemoryDocstore(
        {str(i): Document(page_content=documents[i], metadata={"source": "arxiv"}) for i in range(len(documents))}
    )


    # 3. Load Fine-Tuned Embedding Model


    print("Loading fine-tuned sentence transformer model...")
    embeddings = HuggingFaceEmbeddings(model_name="./fine_tuned_model")


    # 4. Build the Vectorstore


    print("Building vectorstore...")
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    print("Vectorstore is ready!")
    return vectorstore


