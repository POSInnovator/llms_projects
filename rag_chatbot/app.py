import streamlit as st
from src.embeddings import process_file
from src.chat import create_chain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: black;
    }
    .stFileUploader {
        padding: 2rem;
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        background-color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: white;
    }
    .assistant-message {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📚 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload your documents",
        type=["pdf", "txt", "csv", "docx"],
        help="Supported formats: PDF, TXT, CSV, DOCX"
    )
    
    if uploaded_file and not st.session_state.chain:
        with st.spinner("Processing document..."):
            vectorstore = process_file(uploaded_file)
            st.session_state.chain = create_chain(vectorstore)
            st.success("Document processed successfully!")

# Main chat interface
st.title("🤖 RAG Chatbot")

# Display chat messages
for message in st.session_state.messages:
    with st.container():
        st.markdown(f"""
        <div class="chat-message {'user-message' if message['role'] == 'user' else 'assistant-message'}">
            <div>
                <b>{'You' if message['role'] == 'user' else '🤖 Assistant'}</b>
                <br/>
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about your document"):
    if not st.session_state.chain:
        st.error("Please upload a document first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            # Get response from chain
            response = st.session_state.chain({"question": prompt})
            answer = response['answer']
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Rerun to update chat display
        st.rerun()