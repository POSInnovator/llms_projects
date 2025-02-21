import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "faiss_db")


# Define the embedding model 
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Load the existing vector store with embedding function
db_faiss = FAISS.load_local(persistent_directory, embeddings=embeddings, allow_dangerous_deserialization=True)


print(f"Number of documents in FAISS: {(db_faiss.index.ntotal)}")


# Define the user query
query = "Who is Odysseus wife. Just name her if you find."

# Retrieve the relevant documents from FAISS
retriever = db_faiss.as_retriever(
    search_type = 'similarity_score_threshold',
    search_kwargs = {
        'k':2,
        'score_threshold': 0.1,
    }
)

relevant_docs = retriever.invoke(query)
print(f'relevant_docs=={relevant_docs}')

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")