# schemas.py
from typing import Optional
from pydantic import BaseModel, ConfigDict


# Base fields shared across creation and responses
class WaterChemistryBase(BaseModel):
    id_orig: Optional[int] = None
    label: Optional[str] = None
    farm: Optional[str] = None
    doc: Optional[str] = None
    type: Optional[str] = None
    month: Optional[str] = None
    year: Optional[int] = None
    date: Optional[str] = None
    conductivity_us: Optional[float] = None
    ammonium_nh4: Optional[float] = None
    nh4_n_mg_l: Optional[float] = None
    nitrate_no3: Optional[str] = None
    no3_n_mg_l: Optional[float] = None
    nitrite_no2: Optional[str] = None
    no2_n_mg_l: Optional[float] = None
    phosphate_mg_l: Optional[str] = None
    sodium_mg_l: Optional[float] = None
    chloride_mg_l: Optional[float] = None
    calcium_mg_l: Optional[float] = None
    doc_mg_l: Optional[float] = None
    toc_mg_l: Optional[float] = None
    tc_f: Optional[float] = None
    tc_uf: Optional[float] = None
    ic_f: Optional[float] = None
    ic_uf: Optional[float] = None
    tn_f: Optional[float] = None
    tn_uf: Optional[float] = None
    particlesize_um: Optional[float] = None
    absorbance_254nm: Optional[str] = None


# Used for POST /water-chemistry (Client doesn't provide the primary key 'id')
class WaterChemistryCreate(WaterChemistryBase):
    pass


# Used for PATCH /water-chemistry/{id} (all fields optional)
class WaterChemistryUpdate(WaterChemistryBase):
    pass


# Used for API responses (GET requests)
class WaterChemistryResponse(WaterChemistryBase):
    id: int

    # For Pydantic v2:
    model_config = ConfigDict(from_attributes=True)

    # Note: If still on Pydantic v1, use:
    # class Config:
    #     orm_mode = True