from fastapi import FastAPI, Response, status, HTTPException, Depends, APIRouter
from sqlalchemy.orm import Session
from .. import models, schemas, utils, database

router = APIRouter(
    prefix="/users",
    tags=['Users']
)

@router.post("/", status_code=status.HTTP_201_CREATED, response_model=schemas.PatientOut)
def create_patient(patient: schemas.PatientCreate, db: Session = Depends(database.get_db)):

    # 1. Check if email already exists
    existing_patient = db.query(models.Patient).filter(models.Patient.email == patient.email).first()
    if existing_patient:
        raise HTTPException(status_code=400, detail="Email already registered")

    # 2. Hash the password
    hashed_password = utils.hash(patient.password)
    patient.password = hashed_password

    # 3. Create new Patient record
    new_patient = models.Patient(**patient.model_dump())
    
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)

    return new_patient