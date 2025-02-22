import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_dir = os.path.join(db_dir, "faiss_db_firecrawl")

def create_vector_store():

    # Get the Firecrawl api key
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set")
    
    # Crawl the website 
    print(f'\n-- Begin Crawling --')
    loader = FireCrawlLoader(
        api_key=api_key,
        url='https://www.fossil.com/en-us/bags/',
        mode='scrape'
    )
    docs = loader.load()
    print(f'\n-- End Crawling --')

    # Convert metadata values to strings if they are lists
    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    # Split the crawled content into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(split_docs)}")
    print(f"Sample chunk:\n{split_docs[0].page_content[:100]}\n")

    # Create embeddings for the document chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create and persist the vector store with the embeddings
    print(f"\n--- Creating vector store in {persistent_dir} ---")
    db = FAISS.from_documents(
        split_docs, embeddings
    )
    db.save_local(persistent_dir)
    print(f"--- Finished creating vector store in {persistent_dir} ---")

# Check if the FAISS vector store already exists
if not os.path.exists(persistent_dir):
    create_vector_store()
else:
    print(
        f"Vector store {persistent_dir} already exists. No need to initialize.")
    
# Load the vector store with the embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.load_local(persistent_dir, embeddings, allow_dangerous_deserialization=True)

# Query the vector store
def query_vector_store(query):
    """Query the vector store with the specified question."""
    # Create a retriever for querying the vector store
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # Retrieve relevant documents based on the query
    relevant_docs = retriever.invoke(query)

    # Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


# Define the user's question
query = "Jessie Bucket Shoulder Bag?"

# Query the vector store with the user's question
query_vector_store(query)
    
