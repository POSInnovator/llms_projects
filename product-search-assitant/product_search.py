import os
import streamlit as st
import sqlite3
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq

# Custom CSS for styling
st.markdown("""
    <style>
    .stChatMessage { padding: 12px; border-radius: 10px; margin-bottom: 10px; }
    .stChatMessage.user { background-color: #f0f2f6; color: #000; margin-left: 20%; }
    .stChatMessage.assistant { background-color: #4a90e2; color: #fff; margin-right: 20%; }
    .stTextInput input { border-radius: 20px; padding: 10px; }
    .stButton button { border-radius: 20px; background-color: #4a90e2; color: white; padding: 10px 20px; border: none; }
    .sidebar .stTextInput input { border-radius: 10px; padding: 8px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar: API Key Input & Reset Button
with st.sidebar:
    st.header("🔑 API Configuration")
    open_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    groq_api_key = st.text_input("Enter your Groq API key:", type="password")
    
    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.experimental_rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Database connection
conn = sqlite3.connect("products.db")
cursor = conn.cursor()

# Load product data and create embeddings
if "vectors" not in st.session_state and open_api_key and groq_api_key:
    st.session_state.embeddings = OpenAIEmbeddings()
    cursor.execute("SELECT * FROM products")
    products = cursor.fetchall()
    documents = [
    f"ID: {product_id}, Name: {name}, Description: {desc}, Category: {category}, Price: {price} {currency}, URL: {url}" 
    for product_id, name, desc, category, price, currency, url in products
    ]
    st.session_state.vectors = FAISS.from_texts(documents, st.session_state.embeddings)
    st.success("✅ Product database loaded!")

# Define system prompt for product search
system_prompt = """
    You are a product search assistant. Answer user questions based on the product database.
    If a question is outside the product domain, politely decline to answer. DO NOT offer to add or edit any data.
    
    {context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

st.title("🛒 Product Search Assistant")
st.caption("Find products from the database using natural language queries.")

if open_api_key:
    os.environ['OPENAI_API_KEY'] = open_api_key
    os.environ['GROQ_API_KEY'] = groq_api_key
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    if query := st.chat_input("Search for a product..."):
        st.session_state.memory.chat_memory.add_user_message(query)
        st.session_state.messages.append({"role": "user", "content": query, "avatar": "👤"})

        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        # Retrieve product matches
        if "vectors" in st.session_state:
            retriever = st.session_state.vectors.as_retriever()
            document_chain = create_stuff_documents_chain(llm, qa_prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            response = retrieval_chain.invoke({'input': query, 'chat_history': st.session_state.memory.chat_memory.messages})
            answer = response.get("answer", "I couldn't find a matching product. Please refine your query.")

            st.session_state.memory.chat_memory.add_ai_message(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer, "avatar": "🤖"})

            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(answer)

else:
    st.error("⚠️ Please enter a valid OpenAI API key in the sidebar.")

conn.close()
