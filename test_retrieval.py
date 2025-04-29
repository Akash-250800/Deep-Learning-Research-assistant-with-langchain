# test_retrieval.py

from connect_faiss_langchain import load_vectorstore
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 1. Load the Vectorstore


print("Loading vectorstore...")
vectorstore = load_vectorstore()

# 2. Set up LLM and RetrievalQA


print("Setting up LLM and QA chain...")
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

print("Ready to chat!")


# 3. Ask Questions


while True:
    query = input("\nAsk a research question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    try:
        result = qa_chain.invoke({"query": query})
        print("\nAnswer:", result["result"])
    except Exception as e:
        print(f"Error during answering: {e}")

