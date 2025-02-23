import csv, os, json, re
from langchain_groq import ChatGroq
from crawl import query_vector
from logger import Logger


# Create a logger instance
log = Logger(name="create-product-llm", log_file="application.log").get_logger()

# Define the LLM model
llm = ChatGroq(model_name="Gemma2-9b-It")


def process_data_with_llm(data):
    """
    Use LLM to extract structured information from retrieved vector DB data.
    """
    prompt = f"""
    Extract structured product details from the following data:
    {data}
    
    Return a CSV-formatted output with the following columns. Only include column name and data nothing else.
    product_id, category, description.
    """
    
    response = llm.invoke(prompt)
    return response

def save_to_csv(data, output_file="output1.csv"):
    """
    Save extracted product data to a CSV file.
    """
    fieldnames = ["product_id", "category", "description"]
    
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()      
        for row in data:
            if isinstance(row, tuple):  
                row = dict(zip(fieldnames, row))  # Convert tuple to dictionary
            
            if isinstance(row, dict):  
                writer.writerow(row)  # Write only if it's a dictionary


def main():
    """
    Main function to retrieve data from vector DB, process it with LLM, and save to CSV.
    """
    log.info(f"Starting to save data in output.csv")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "db", "product_vector")  # Update this based on your directory
    context = "Retrieve all product details"  # No specific query, fetch all
    
    # Query FAISS Vector DB
    documents = query_vector(context, db_dir)  # Assuming db is managed internally
    
    # Process with LLM
    extracted_data = process_data_with_llm(documents)
    # Save to CSV
    save_to_csv(extracted_data)
    log.info(f"Data saved to output.csv")

if __name__ == "__main__":
    main()
