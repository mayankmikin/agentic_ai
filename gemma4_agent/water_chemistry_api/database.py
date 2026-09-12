# database.py
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL = "sqlite:///water_chemistry.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    },  # Required for SQLite in multi-threaded FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Reflect existing tables
Base = automap_base()
Base.prepare(autoload_with=engine)

WaterChemistry = Base.classes.water_chemistry


# FastAPI Dependency: provides a scoped DB session per request
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()