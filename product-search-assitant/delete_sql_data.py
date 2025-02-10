import sqlite3

# Connect to SQLite database
connection = sqlite3.connect("products.db")
cursor = connection.cursor()

# Correct DELETE statement
cursor.execute("DELETE FROM PRODUCTS")  # No "*" needed

# Commit changes and close the connection
connection.commit()
connection.close()
