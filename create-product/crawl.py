import os, json
from dotenv import load_dotenv
from logger import Logger
from langchain_community.document_loaders import FireCrawlLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# Load the environment
load_dotenv()

# Create a logger instance
log = Logger(name="create-product-crawl", log_file="application.log").get_logger()

# Crawl the URL 
def crawl_site(url: str):
    
    # Logging 
    log.info(f'Started the site crawl : {url}')

    # Get the Firecrawl api key
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        log.error('Missing FIRECRAWL_API_KEY')
        raise ValueError('Firecrawl key missing: FIRECRAWL_API_KEY')
    
    # Crawl the website & load
    loader = FireCrawlLoader(
        api_key=api_key,
        url=url,
        mode='scrape'
    )
    docs = loader.load()

    # Verify the scrapping was successful 
    if len(docs) > 0:
        log.info(f'Site crawl ended successfully')
        return docs
    else:
        log.error("Couldn't scrape the provided URL")
        raise ValueError("Couldn't scrape the provided URL")
    
   

# Store the data in the vector DB
def store_vector(documents: list, db_dir :str):

    # Create embeddings for the document chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
      
    if not os.path.exists(db_dir):

        # Logging 
        log.info(f'Store vector initializing...')

        # Split the crawled content into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_docs = text_splitter.split_documents(documents)
        #print(f"Sample chunk:\n{split_docs[0].page_content[:50]}\n")

         # Create and persist the vector store with the embeddings
        log.info(f'Storing the data into the FAISS index: {db_dir}')
        db = FAISS.from_documents(split_docs, embeddings)
        db.save_local(db_dir)
        log.info(f'Storing in the vector DB completed')
        return db

    else: 
        log.info(f'Vector DB is already created so returning the instance')
        db = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
        return db


# Fetch the data from vector DB
def query_vector(context: str, db_dir: str):

    if not os.path.exists(db_dir):
        raise Exception(f"Paht {db_dir} doesn't exist!")
    
    # Create embeddings for the document chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
    
    # Logging
    log.info('Getting the context from vector DB')

    # Create a retriever for querying the vector store
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # Retrieve relevant documents based on the query
    relevant_docs = retriever.invoke(context)
    #print(relevant_docs)
    log.info(f'Context retrieved : {len(relevant_docs)}')
    return relevant_docs

# Main Method
if __name__ == "__main__":
    context = "Harlow Three-Hand Black Leather Watch "
    db_dir = "/Users/chandantiwari/chandan/ml_projects/llm_projects/llmprojects/create-product/db/product_vector"
    docs = crawl_site('https://www.fossil.com/en-us/watches/womens-watches/')
    db = store_vector(docs, db_dir)
    query_vector(context, db_dir)
