from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import PyPDF2
import docx
import pandas as pd
import io

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_file(file):
    text = ""
    if file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.name.endswith('.txt'):
        text = file.getvalue().decode('utf-8')
    elif file.name.endswith('.docx'):
        doc = docx.Document(io.BytesIO(file.getvalue()))
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file)
        text = df.to_string()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    return vectorstore