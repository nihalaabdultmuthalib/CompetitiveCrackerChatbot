import streamlit as st
import json
from datetime import datetime
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA


# ============================================================
# SAVE CHAT TO JSON
# ============================================================

HISTORY_FILE = Path(__file__).resolve().parent / "chat_history.json"


def save_chat(question, answer):

    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            history = json.load(file)

    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append({
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    with HISTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, ensure_ascii=False)

    return HISTORY_FILE


def load_chat_history():
    """Convert saved JSON records into Streamlit chat messages."""
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            history = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    messages = []
    for item in history:
        question = item.get("question")
        answer = item.get("answer")
        if question:
            messages.append({"role": "user", "content": question})
        if answer:
            messages.append({"role": "assistant", "content": answer})
    return messages


def clear_chat_history():
    with HISTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump([], file, indent=4)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Competitive Cracker AI",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 20px;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

h1 {
    color: #1E88E5;
    text-align: center;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712027.png",
        width=120
    )

    st.title("Competitive Cracker")

    st.write("""
Welcome to the AI Assistant.

This chatbot can answer questions related to

✅ KTET

✅ CTET

✅ Kerala SET

✅ NET

✅ PSC

✅ Banking

✅ SSC

✅ UPSC

using the knowledge base.
""")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []
        clear_chat_history()

        st.rerun()


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Competitive Cracker AI Chatbot")


# ============================================================
# LOAD CHATBOT
# ============================================================

@st.cache_resource
def load_chatbot():

    # Embedding model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434"
    )

    # Load existing Chroma database
    vector_db = Chroma(
        persist_directory="./chroma_db_data",
        embedding_function=embeddings
    )

    # Retriever
    retriever = vector_db.as_retriever()

    # Llama 3
    llm = Ollama(
        model="llama3",
        base_url="http://127.0.0.1:11434"
    )

    # Retrieval QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa


# Load chatbot
qa = load_chatbot()


# ============================================================
# STREAMLIT CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = load_chat_history()


# Display previous messages in current session

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask any question about Competitive Cracker"
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Save user question in current Streamlit session
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            # TEST STEP 1
            st.write("STEP 1: Starting AI response...")


            # ------------------------------------------------
            # Retrieve relevant documents and generate answer
            # ------------------------------------------------

            result = qa.invoke(
                {
                    "query": question
                }
            )


            # TEST STEP 2
            st.write("STEP 2: AI response received")


            # ------------------------------------------------
            # Extract answer
            # ------------------------------------------------

            answer = result["result"]


            # TEST STEP 3
            st.write("STEP 3: Saving chat...")


            # ------------------------------------------------
            # Save question + answer to JSON
            # ------------------------------------------------

            saved_file = save_chat(
                question,
                answer
            )


            # TEST STEP 4
            st.write(f"STEP 4: Chat saved successfully to {saved_file.name}")


            # ------------------------------------------------
            # Display answer
            # ------------------------------------------------

            st.markdown(answer)


    # --------------------------------------------------------
    # Save assistant answer in Streamlit session
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
