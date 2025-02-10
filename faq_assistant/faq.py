import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


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
    groq_api_key = st.text_input("Enter your Groq API key:", type="password")
    open_api_key = st.text_input("Enter your OpenAI API key:", type="password")

    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.experimental_rerun()

    # Upload the Policy Files
    st.title("📚 Upload Policy Files")
    uploaded_file = st.file_uploader("Upload your document", type=["pdf"], help="Supported formats: PDF")

# Initialize session state for chat memory and document processing
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Process Uploaded File
if uploaded_file and "vectors" not in st.session_state:
    st.session_state.embeddings = OpenAIEmbeddings()

    # Save file temporarily for processing
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process the document
    st.session_state.loader = PyPDFLoader("temp.pdf")
    st.session_state.docs = st.session_state.loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs[:50])

    if not st.session_state.final_documents:
        st.error("No valid documents loaded. Please try again.")
    else:
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("📖 FAQ Document Processed Successfully!")

# Define the system prompt for context-aware responses
system_prompt = """ 
    You are an FAQ assistant. Answer user questions based on the uploaded FAQ document. 
    If a question is outside the FAQ, politely decline to answer.
    Maintain conversation history for better context. 
    \n\n {context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Streamlit UI
st.title("💬 Personalized FAQ Assistant")
st.caption("Get answers from your uploaded FAQ documents.")

# Main Chat Interface
if groq_api_key and open_api_key:
    os.environ['OPENAI_API_KEY'] = open_api_key
    os.environ['GROQ_API_KEY'] = groq_api_key

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    if uploaded_file:
        if prompt := st.chat_input("Ask a question based on the FAQ..."):
            
            # Append user input to memory
            st.session_state.memory.chat_memory.add_user_message(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👤"})

            # Display user input
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # Generate response using retrieval chain
            if "vectors" in st.session_state:
                retriever = st.session_state.vectors.as_retriever()
                document_chain = create_stuff_documents_chain(llm, qa_prompt)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                response = retrieval_chain.invoke({'input': prompt, 'chat_history': st.session_state.memory.chat_memory.messages})
                answer = response.get("answer", "I'm not sure about that. Please try rephrasing.")

                # Append assistant response to memory
                st.session_state.memory.chat_memory.add_ai_message(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer, "avatar": "🤖"})

                # Display assistant response
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(answer)

else:
    st.error("⚠️ Please enter valid API keys in the sidebar.")
