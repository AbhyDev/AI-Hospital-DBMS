from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from .. import database, schemas, models, utils, oauth2

router = APIRouter(tags=['Authentication'])

@router.post('/login', response_model=schemas.Token)
def login(user_credentials: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    
    # 1. Find patient by email (username field in OAuth2 form = email)
    patient = db.query(models.Patient).filter(
        models.Patient.email == user_credentials.username
    ).first()

    # 2. Check if patient exists
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid Credentials"
        )

    # 3. Verify password
    if not utils.verify(user_credentials.password, patient.password):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid Credentials"
        )

    # 4. Create Token (We use patient_id as the 'user_id' in token)
    access_token = oauth2.create_access_token(data={"user_id": patient.patient_id})

    return {"access_token": access_token, "token_type": "bearer"}