import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.chains import LLMChain

# Custom CSS for styling
st.markdown("""
    <style>
    .stChatMessage {
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: #f0f2f6;
        color: #000;
        margin-left: 20%;
    }
    .stChatMessage.assistant {
        background-color: #4a90e2;
        color: #fff;
        margin-right: 20%;
    }
    .stTextInput input {
        border-radius: 20px;
        padding: 10px;
    }
    .stButton button {
        border-radius: 20px;
        background-color: #4a90e2;
        color: white;
        padding: 10px 20px;
        border: none;
    }
    .sidebar .stTextInput input {
        border-radius: 10px;
        padding: 8px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for API key input and Reset button
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("Enter your Groq API key:", type="password")

    # Reset button to clear chat history and session state
    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

# Title of the app
st.title("💬 LLM Chat with History")
st.caption("A chat app that remembers past conversations.")

# Check if API key is provided
if api_key:
    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Define system prompt
    system_prompt = (
        "You are a helpful AI assistant. "
        "Use the chat history to provide relevant responses. "
        "Keep your responses concise."
    )

    # Define prompt template
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Create a ConversationBufferMemory instance
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create a chain with the LLM, the prompt template, and memory
    llm_chain = LLMChain(llm=llm, prompt=qa_prompt, memory=memory)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Type your message here..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👤"})
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Generate response from LLM using the chain with memory
    if api_key:
        # Run the chain with the input prompt, chat history, and memory
        response = llm_chain.run(input=prompt)
    else:
        response = "API key is missing. Please enter your API key in the sidebar."
    
    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "🤖"})
    
    # Display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)
