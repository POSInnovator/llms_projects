import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
curr_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(curr_dir, "db", "faiss_email-opt-in")
db_dir = os.path.join(curr_dir,"db")

# WebBaseLoader loads web pages and extracts their content
urls = ['https://docs.newstore.net/developers/configuration/associate-app-config/config-aa-email-opt-in/']

# Load the loader for the web content
loader = WebBaseLoader(urls)
documents = loader.load()

# Split the document into chunks 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content[:2000]}\n")

# Create embeddings for the document chunks
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create and persist the vector store with the embeddings
if not os.path.exists(persistent_dir):
    print(f'\nThe vector DB {persistent_dir} doesn not exist so initializing!')
    db = FAISS.from_documents(documents=docs, embedding=embeddings)
    db.save_local(persistent_dir)
else:
    print(f'\nThe vector DB {persistent_dir} already exist so skipping the initialization!')
    db = FAISS.load_local(persistent_dir, embeddings, allow_dangerous_deserialization=True)

# Create the DB retriver
retriver = db.as_retriever(
    search_type = "similarity_score_threshold",
            search_kwargs = {
                "k": 2, 
                "score_threshold": 0.1
            },
)

# User query and retrieve context
context_query = 'EMail Opt-in'
relevant_docs = retriver.invoke(context_query)
context = []

# Display the relevant results with metadata
#print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    context = '\n'.join([doc.page_content])
    #print(f"\nDocument {i}:\n{doc.page_content[:50]}\n")

# Create a ChatOpenAI model
prompt = f"""
         Use the following context to answer the question. DO NOT add anything additional outside the context: 
         {context}

         Query : What is the process for email opt-in?
         Answer: 
         """

llm = ChatGroq(model_name="Gemma2-9b-It")
#print(f'\nprompt======={prompt}')
result = llm.invoke(prompt)
print(f'\n--- Summary ----\n{result.content}')