import sqlite3

def query_portfolio_db(query_type: str, parameter: str) -> str:
    """
    Queries the local SQLite database for portfolio and asset data.
    query_type options: 'balance' (requires username), 'ticker' (requires stock symbol)
    """
    conn = sqlite3.connect("portfolio.db")
    cursor = conn.cursor()

    try:
        if query_type == "balance":
            cursor.execute("SELECT balance, currency FROM users WHERE username = ?", (parameter,))
            row = cursor.fetchone()
            if row:
                return f"User {parameter} has a balance of {row[0]} {row[1]}."
            return f"User {parameter} not found."

        elif query_type == "ticker":
            cursor.execute("SELECT price, currency FROM assets WHERE ticker = ?", (parameter.upper(),))
            row = cursor.fetchone()
            if row:
                return f"The current price of {parameter.upper()} is {row[0]} {row[1]}."
            return f"Ticker {parameter.upper()} not found in database."

        else:
            return "Error: Invalid query_type provided."

    except sqlite3.Error as e:
        return f"Database error occurred: {str(e)}"
    finally:
        conn.close()


