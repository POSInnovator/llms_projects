import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

#Fetch env
load_dotenv()

# Load the path of the FAISS Index
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "faiss_with_metadata")


# Define the embedding model 
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Load the existing vector store with embedding function
db_faiss = FAISS.load_local(persistent_directory, embeddings=embeddings, allow_dangerous_deserialization=True)

# User Query 
query = 'Who is Jane Austen?'

# Set the retriver details
retriver = db_faiss.as_retriever(
    search_type = 'similarity_score_threshold',
    search_kwargs = {
        'k':2,
        'score_threshold': 0.1,
    }
)

# Display the relevant results with metadata
relevant_docs = retriver.invoke(query)
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata['source']}\n")

