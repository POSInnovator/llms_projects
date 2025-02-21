import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

#Loading env
load_dotenv()

# Define the directory containing the text files and the persistent directory 
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory  = os.path.join(db_dir, "faiss_with_metadata")
print(f'\nbooks_dir = {books_dir}\npersistent_directory = {persistent_directory}')


# Check if the FAISS DB vactor already exist
if not os.path.exists(persistent_directory):
   print('\npersistent directory does not exist. Initializing vector store...')

   # Ensure that the books_dir exist
   if not os.path.exists(books_dir):
       raise FileNotFoundError(
           f'The directory doesn not exist\n{books_dir} '
       )
   
   # List all the text files in the directory 
   books_files = [ file for file in os.listdir(books_dir) if file.endswith('.txt')]
   print(f'\nbooks_files\n{books_files}')

   # Read the text content from each file and store it was meta data
   documents = [] 
   for book_file in books_files:
       file_path = os.path.join(books_dir, book_file)
       loader = TextLoader(file_path)
       book_docs = loader.load()
       
       # Add the meta data for each book docs
       for doc in book_docs: 
           doc.metadata = {'source': book_file}
           #print(f'\doc\n{doc}')
           documents.append(doc)
           
   #print(f'\nDocument\n{documents}')

   #Split the documents into chunks
   text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
   docs = text_splitter.split_documents(documents)
   print('\n--- Document Chunks Information ---')
   print(f'Number of document chunks = {len(docs)}')

   #Create embeddings 
   print('\n-- Creating Embeddings --')
   embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
   print(f'\n-- Finished creating embeddings --')

   # Create vectorstore and persist it 
   print("\n--- Creating and persisting vector store ---")
   db_faiss = FAISS.from_documents(docs, embeddings)
   # Now save the FAISS index to a persistent directory
   db_faiss.save_local(persistent_directory)
   print("\n--- Finished creating and persisting vector store ---")


else:
    print('\nVector store already exists. No need to initialize')