import sqlite3

def create_and_seed_db():
    # Connect to sqlite (this will create portfolio.db if it doesn't exist)
    conn = sqlite3.connect("portfolio.db")
    cursor = conn.cursor()

    print("Creating tables...")
    # 1. Create Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            balance REAL NOT NULL,
            currency TEXT NOT NULL
        )
    """)

    # 2. Create Assets Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE NOT NULL,
            price REAL NOT NULL,
            currency TEXT NOT NULL
        )
    """)

    print("Seeding records...")
    # Seed user balances
    users_data = [
        ('mayank', 15000.50, 'USD'),
        ('alice', 2850.00, 'EUR'),
        ('bob', 750.25, 'GBP')
    ]
    cursor.executemany("INSERT OR IGNORE INTO users (username, balance, currency) VALUES (?, ?, ?)", users_data)

    # Seed asset prices
    assets_data = [
        ('AAPL', 185.50, 'USD'),
        ('GOOG', 175.20, 'USD'),
        ('MSFT', 420.10, 'USD'),
        ('AMZN', 180.05, 'USD')
    ]
    cursor.executemany("INSERT OR IGNORE INTO assets (ticker, price, currency) VALUES (?, ?, ?)", assets_data)

    # Commit changes and close
    conn.commit()
    conn.close()
    print("Database 'portfolio.db' successfully created and populated!")

if __name__ == "__main__":
    create_and_seed_db()

    ### python3 seed_db.py