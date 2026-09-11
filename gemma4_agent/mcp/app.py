from fastapi import FastAPI, HTTPException

import sqlite3

from models import BookCreate

app = FastAPI()

@app.get("/")
def read_root():
 return {"message": "Welcome to the CRUD API"}

@app.post("/books/")
def create_book_endpoint(book: BookCreate):
 book_id = create_book(book)
 return {"id": book_id, **book.dict()}

@app.get("/books/")
def get_books():
 return read_books()

def create_connection():
 connection = sqlite3.connect("books.db")
 return connection

def create_book(book: BookCreate):
 connection = create_connection()
 cursor = connection.cursor()
 cursor.execute("INSERT INTO books (title, author) VALUES (?, ?)", (book.title, book.author))
 connection.commit()
 connection.close()

def read_books():
  connection = create_connection()
  cursor = connection.cursor()
  try:
   cursor.execute("select * from books")
   rows = cursor.fetchall()
   connection.commit()
   connection.close()
   return [{"id": r[0], "title": r[1], "Author": r[2]} for r in rows]
  except Exception as e:
   return HTTPException(status_code=400, detail="Unable to read {e}")

