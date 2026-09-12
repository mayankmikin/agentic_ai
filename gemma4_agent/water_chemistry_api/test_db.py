from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, create_model
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

# 1. Setup Engine & Automap
DATABASE_URL = "sqlite:///water_chemistry.db"
engine = create_engine(DATABASE_URL)

Base = automap_base()
Base.prepare(autoload_with=engine)
WaterChemistry = Base.classes.water_chemistry

# 2. Extract column types reliably
fields = {}
for column in WaterChemistry.__table__.columns:
    # Try getting the Python type directly from SQLAlchemy type engine
    try:
        py_type = column.type.python_type
    except (NotImplementedError, AttributeError):
        # Fallback for custom or unconventional dialect types
        col_type_str = str(column.type).split("(")[0].strip().upper()
        type_mapping = {
            "INTEGER": int,
            "REAL": float,
            "FLOAT": float,
            "NUMERIC": float,
            "TEXT": str,
            "VARCHAR": str,
            "DATETIME": str,
            "BOOLEAN": bool,
        }
        py_type = type_mapping.get(col_type_str, Any)

    # Required for primary key, Optional with default None for others
    if column.primary_key:
        fields[column.name] = (py_type, ...)
    else:
        fields[column.name] = (Optional[py_type], None)


# 3. Use an explicit Base class to guarantee zero metaclass conflicts
class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


WaterChemistrySchema = create_model(
    "WaterChemistrySchema", __base__=BaseSchema, **fields
)

# 4. Fetch and validate
with Session(engine) as session:
    record = session.query(WaterChemistry).first()
    if record:
        pydantic_data = WaterChemistrySchema.model_validate(record)
        print("✅ Successfully validated record:")
        print(pydantic_data.model_dump())
    else:
        print("Table is empty.")