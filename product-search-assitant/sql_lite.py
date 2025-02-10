import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
connection = sqlite3.connect("products.db")

# Create a cursor object to execute SQL commands
cursor = connection.cursor()

# Create the PRODUCTS table
table_info = """
CREATE TABLE IF NOT EXISTS PRODUCTS (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    category VARCHAR(50),
    price DECIMAL(10, 2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',
    url VARCHAR(255)
);
"""
cursor.execute(table_info)

# Insert product records into the table
products_data = [
    ("Elegant Dress", "A timeless black dress for formal occasions", "Women's Clothing", 199.99, "USD", "https://example.com/elegant-dress"),
    ("Leather Jacket", "Premium quality leather jacket for men", "Men's Clothing", 299.99, "USD", "https://example.com/leather-jacket"),
    ("Silk Scarf", "Handwoven silk scarf with floral patterns", "Accessories", 49.99, "USD", "https://example.com/silk-scarf"),
    ("Running Shoes", "Lightweight running shoes for athletes", "Footwear", 89.99, "USD", "https://example.com/running-shoes"),
    ("Smartwatch", "Latest smartwatch with health tracking features", "Electronics", 199.99, "USD", "https://example.com/smartwatch"),
]

# Insert each product into the table
for product in products_data:
    cursor.execute('''
        INSERT INTO PRODUCTS (name, description, category, price, currency, url)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', product)

# Display all the records
print("The inserted records are:")
data = cursor.execute('''SELECT * FROM PRODUCTS''')
for row in data:
    print(row)

# Commit changes and close the connection
connection.commit()
connection.close()