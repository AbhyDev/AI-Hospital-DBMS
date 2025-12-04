from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

# --- Auth Schemas (Copied from Fastend User) ---
class PatientCreate(BaseModel):
    email: EmailStr
    password: str
    name: str
    age: int
    gender: str

class PatientLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    id: Optional[str] = None

# --- Response Schemas (For returning data to frontend) ---
class PatientOut(BaseModel):
    patient_id: int
    email: EmailStr
    name: str
    created_at: datetime

    class Config:
        from_attributes = True

# --- Medical Schemas (Optional, but good for Tool validation) ---
class ConsultationOut(BaseModel):
    consultation_id: int
    status: str
    started_at: datetime