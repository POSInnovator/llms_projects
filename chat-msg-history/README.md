# LLM Chat with History

This is a **Streamlit** application that allows users to have a chat with an AI assistant, which remembers past conversations. The assistant uses **Groq's API** for generating responses, and the conversation context is maintained using LangChain's **ConversationBufferMemory** to ensure relevant and consistent responses based on previous exchanges.

## Features

- **Chat History**: The assistant remembers all prior exchanges in the conversation and generates responses based on the entire chat history.
- **Custom Styling**: The chat interface is styled with custom CSS to make the chat messages visually appealing.
- **Reset Button**: The user can reset the chat and clear the session history with a click of a button.
- **Persistent Memory**: The app utilizes LangChain's `ConversationBufferMemory` to store and manage conversation history automatically.
  
## Technologies

- **Streamlit**: Framework used to create the user interface.
- **LangChain**: Library used for prompt engineering, memory management, and LLM integration.
- **Groq API**: Used to invoke the AI model for generating responses.
  
## Requirements

- **Groq API Key**: Required to authenticate with the Groq API and generate responses.
- **Streamlit**: To run the app. Install using `pip install streamlit`.
- **LangChain**: To handle prompt templates and memory. Install using `pip install langchain`.

## Setup

1. Clone this repository or download the code.
2. Install the required dependencies:
   ```bash
   pip install streamlit langchain langchain-groq
