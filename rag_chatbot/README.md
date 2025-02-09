# RAG Chatbot

A professional conversational chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on uploaded documents.

## Features

- Upload and process multiple document types (PDF, TXT, CSV, DOCX)
- Conversational memory to maintain context
- Professional UI with Streamlit
- Uses FAISS for efficient vector storage
- Powered by Groq's Mixtral-8x7B model

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables:
- Copy `.env.example` to `.env`
- Add your Groq API key to `.env`

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload your document using the sidebar
2. Wait for the document to be processed
3. Start asking questions about your document
4. The chatbot will provide relevant answers based on the document content

## Architecture

- `app.py`: Main Streamlit application
- `src/embeddings.py`: Document processing and embedding generation
- `src/chat.py`: Chat chain setup and LLM configuration