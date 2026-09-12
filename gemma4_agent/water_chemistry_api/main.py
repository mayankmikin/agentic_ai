# main.py
from typing import List
from database import WaterChemistry, get_db
from fastapi import Depends, FastAPI, HTTPException, Query, status
from typing import List, Optional
from schemas import (
    WaterChemistryCreate,
    WaterChemistryResponse,
    WaterChemistryUpdate,
)
from sqlalchemy.orm import Session

app = FastAPI(
    title="Water Chemistry API",
    description="API for managing water chemistry lab measurements.",
    version="1.0.0",
)


# 1. READ ALL (with pagination & optional farm filtering)

@app.get(
    "/water-chemistry",
    response_model=List[WaterChemistryResponse],
    tags=["Water Chemistry"],
)
def get_all_records(
    id_orig: Optional[int] = Query(
        None, description="Filter records by original ID"
    ),
    farm: Optional[str] = Query(None, description="Filter records by farm"),
    skip: int = Query(0, ge=0, description="Offset"),
    limit: int = Query(50, ge=1, le=200, description="Page size"),
    db: Session = Depends(get_db),
):
    query = db.query(WaterChemistry)

    # Filter by id_orig if provided in the query string
    if id_orig is not None:
        query = query.filter(WaterChemistry.id_orig == id_orig)

    # Filter by farm if provided
    if farm:
        query = query.filter(WaterChemistry.farm == farm)

    return query.offset(skip).limit(limit).all()


# 2. READ ONE BY ID
@app.get(
    "/water-chemistry/{record_id}",
    response_model=WaterChemistryResponse,
    tags=["Water Chemistry"],
)
def get_record(record_id: int, db: Session = Depends(get_db)):
    record = (
        db.query(WaterChemistry)
        .filter(WaterChemistry.id == record_id)
        .first()
    )
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record with ID {record_id} not found.",
        )
    return record


# 3. CREATE RECORD
@app.post(
    "/water-chemistry",
    response_model=WaterChemistryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Water Chemistry"],
)
def create_record(
    payload: WaterChemistryCreate, db: Session = Depends(get_db)
):
    # Convert Pydantic model to a dict, excluding unset fields if needed
    data = payload.model_dump()  # In Pydantic v1, use: payload.dict()
    db_record = WaterChemistry(**data)

    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record


# 4. UPDATE RECORD
@app.patch(
    "/water-chemistry/{record_id}",
    response_model=WaterChemistryResponse,
    tags=["Water Chemistry"],
)
def update_record(
    record_id: int,
    payload: WaterChemistryUpdate,
    db: Session = Depends(get_db),
):
    record = (
        db.query(WaterChemistry)
        .filter(WaterChemistry.id == record_id)
        .first()
    )
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record with ID {record_id} not found.",
        )

    # Update only provided values
    update_data = payload.model_dump(
        exclude_unset=True
    )  # In Pydantic v1, use: payload.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(record, key, value)

    db.commit()
    db.refresh(record)
    return record


# 5. DELETE RECORD
@app.delete(
    "/water-chemistry/{record_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Water Chemistry"],
)
def delete_record(record_id: int, db: Session = Depends(get_db)):
    record = (
        db.query(WaterChemistry)
        .filter(WaterChemistry.id == record_id)
        .first()
    )
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record with ID {record_id} not found.",
        )

    db.delete(record)
    db.commit()
    return None

