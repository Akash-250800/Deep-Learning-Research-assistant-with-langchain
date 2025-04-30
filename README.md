#  Deep Learning Research Assistant

A Streamlit-powered chatbot built using **LangChain**, **OpenAI**, and **FAISS** to answer research questions from custom documents.
This assistant uses vector similarity search with OpenAI's LLMs for accurate, document-grounded answers.

---

##  Features

-  Context-aware Q&A powered by OpenAI GPT models
-  Retrieval-Augmented Generation using FAISS vector search
-  Supports custom document embeddings (`.pkl` and `.faiss`)
- Simple chat interface with memory (session history)
-  Deployable on [Streamlit Cloud](https://streamlit.io/cloud)

---

##  Live App

> 🔗 [Launch the App](https://your-username-your-repo.streamlit.app)

https://deep-learning-research-assistant-with-langchain-szqnuzxfiyliyr.streamlit.app/

---

##  Project Structure

```bash
├── app.py                     # Streamlit app
├── connect_faiss_langchain.py # Loads FAISS + embeddings
├── build_faiss.py             # (Optional) Generates FAISS index
├── index.faiss                # Vector store index
├── index.pkl
├── docstore.pkl
├── documents.pkl              # Saved input documents
├── .env.example               # Sample for OpenAI API key
├── requirements.txt           # Python dependencies
├── .gitignore

Local Setup
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/Akash-250800/Deep-Learning-Research-assistant-with-langchain.git
cd Deep-Learning-Research-assistant-with-langchain
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Add your OpenAI API key
Create a .env file:

env
Copy
Edit
OPENAI_API_KEY=sk-your-real-openai-key
Never share your real .env file. Only .env.example is tracked in Git.

4. Run the app
bash
Copy
Edit
streamlit run app.py
 Build Your Own FAISS Index (Optional)
Use build_faiss.py to create a new FAISS index from your custom documents.

 Streamlit Cloud Deployment
Push code to GitHub

Go to Streamlit Cloud

Click New App → Connect GitHub → Select app.py

Add your OpenAI key in Secrets:

toml
Copy
Edit
OPENAI_API_KEY = "sk-your-real-key"
Click Deploy

 License
MIT License

 Author
Akash
Connect with me on GitHub

 Show Your Support
If you found this helpful, please star the repo
