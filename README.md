# Competitive Cracker AI Chatbot

An AI-powered chatbot that answers questions about competitive exams using
Retrieval-Augmented Generation (RAG).

## Features
- Answers questions from the Competitive Cracker knowledge base
- Uses text chunking and embeddings
- ChromaDB for vector storage
- Ollama for local LLM inference
- Streamlit user interface

## Tech Stack
- Python
- Streamlit
- LangChain
- ChromaDB
- Ollama
- RAG

## How to Run

### 1. Clone the repository
git clone https://github.com/nihalaabdultmuthalib/CompetitiveCrackerChatbot.git

### 2. Install dependencies
pip install -r requirements.txt

### 3. Make sure Ollama is running

Required models:
- llama3
- nomic-embed-text

### 4. Run the application
streamlit run app.py

The application will open at:
http://localhost:8501
