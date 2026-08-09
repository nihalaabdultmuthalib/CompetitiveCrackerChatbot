import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

# ---------------- PAGE CONFIG ----------------

import streamlit as st

st.set_page_config(
    page_title="Competitive Cracker AI",
    page_icon="🤖",
    layout="wide"
)




# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main{
    padding-top:20px;
}

.stChatMessage{
    border-radius:15px;
    padding:10px;
}

h1{
    color:#1E88E5;
    text-align:center;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=120)

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

        st.session_state.messages=[]

        st.rerun()

# ---------------- TITLE ----------------

st.title("🤖 Competitive Cracker AI Chatbot")

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_chatbot():

    embeddings=OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_db=Chroma(
        persist_directory="./chroma_db_data",
        embedding_function=embeddings
    )

    retriever=vector_db.as_retriever()

    llm=Ollama(
        model="llama3"
    )

    qa=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa

qa=load_chatbot()

# ---------------- CHAT HISTORY ----------------

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ---------------- CHAT INPUT ----------------

question=st.chat_input("Ask any question about Competitive Cracker")

if question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer=qa.run(question)

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )