import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

st.set_page_config(page_title="Mimin CS AI", page_icon="🤖")
load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not GROQ_API_KEY or not DATABASE_URL:
    st.error("WOY! File .env lo mana? API Key atau DB URL kosong tuh!")
    st.stop()

@st.cache_resource
def get_rag_chain():
    # A. Setup Embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # B. Fix Connection String
    connection_string = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+psycopg://")
    
    # C. Connect DB (Pre-Ping)
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="customer_support_vector_db",
        connection=connection_string,
        use_jsonb=True,
        engine_args={
            "pool_size": 1,
            "max_overflow": 0,
            "pool_recycle": 300,
            "pool_pre_ping": True,
            "connect_args": {
                "connect_timeout": 10,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            }
        }
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # D. Setup LLM
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY
    )

    # E. Contextualize Prompt (Benerin Pertanyaan User)
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. 
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    IMPORTANT: Keep the standalone question in the SAME LANGUAGE as the user's latest question."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # F. QA Prompt
    qa_system_prompt = """You are a helpful and friendly Customer Assistant named 'Mimin'.
    Use the following pieces of retrieved context to answer the question.

    RULES:
    1. Answer in **Indonesian Language (Bahasa Indonesia)**, unless the user specifically asks in English.
    2. If the context does not contain the answer, say "Waduh, maaf kak, Mimin gak nemu infonya di catatan." (Don't make up answers).
    3. Keep the answer concise (max 3 sentences) but polite.
    
    Context:
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# --- Session State Management (Memory) ---
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Init Chain
try:
    base_chain = get_rag_chain()
    # Bungkus pake Memory
    conversational_rag_chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
except Exception as e:
    st.error(f"Gagal connect ke Database/Groq. Error: {e}")
    st.stop()

# 3. UI Chatbot
st.title("🤖 Customer Service AI")
st.caption("Mimin siap membantu (kalau datanya ada).")

# Session ID
session_id = "user_demo_session"

# Initialize Chat UI History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan Pesan Lama
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input User
if prompt := st.chat_input("Tanya Mimin..."):
    # 1. Tampilkan User Msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Jawaban
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Mimin lagi buka buku panduan..."):
            try:
                # Panggil RAG Chain
                response = conversational_rag_chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": session_id}}
                )
                answer = response["answer"]
                
                # Simulasi ngetik
                for chunk in answer.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"Error: {e}")
                answer = "Maaf, Mimin lagi pusing (Error System)."
                message_placeholder.markdown(answer)

    # 3. Simpan Bot Msg
    st.session_state.messages.append({"role": "assistant", "content": answer})