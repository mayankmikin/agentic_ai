from pydantic import BaseModel

class BookCreate(BaseModel):
 title: str
 author: str

class Book(BookCreate):
 id: int

