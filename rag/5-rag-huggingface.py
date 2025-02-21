import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
db_dir = os.path.join(current_dir, "db")

# Check if the txt file exist
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f'The file path does nto exist - {file_path}'
    )


# Function to create and persist vector store
def create_vector_store(docs, embeddings, store_name):
    file_path = os.path.join(db_dir, store_name)
    if not os.path.exists(file_path):
        print(f'\n-- Creating vector store {store_name} --')
        db_faiss = FAISS.from_documents(docs, embeddings)
        db_faiss.save_local(file_path)
    else:
        print(f'The {store_name} already exist so skipping it!')

# Function to query from the vector store
def query_vector_store(store_name, query, huggingface_embeddings):
    file_path = os.path.join(db_dir, store_name)
    
    # Retrieve the relevant documents from FAISS  
    if os.path.exists(file_path):

        # Load the existing vector store with embedding function
        db_faiss = FAISS.load_local(file_path, huggingface_embeddings, allow_dangerous_deserialization=True)
        
        # Retrieve from the DB
        retriever = db_faiss.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {
                "k": 2, 
                "score_threshold": 0.1
            },
        )
        relevant_docs = retriever.invoke(query)
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else: 
        print(f'\nThere is no vector db in the path - {file_path}')

#  Read the txt file from the path 
loader = TextLoader(file_path)
documents = loader.load()

# Split the documents into Chunks 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(f'\n-- Document chunks information --')
print(f'Number of document: {len(docs)}')
print(f'Sample chunk:\n{docs[0].page_content[:100]}')


# Hugging Face Transformers
print(f'\n-- Using hugging face transformers --')
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-mpnet-base-v2'
)
create_vector_store(docs, huggingface_embeddings, 'faiss_db_huggingface')
print("-- Embedding demonstrations for Hugging Face completed. --")

# Query a text in the vector DB
query = 'Who is Odysseus?'

# Query vector store
query_vector_store('faiss_db_huggingface', query, huggingface_embeddings)
print("Querying demonstrations completed.")