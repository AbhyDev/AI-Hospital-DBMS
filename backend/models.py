from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Text, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql.expression import text
from .database import Base

# 1. PATIENT (Replaces 'User' from Fastend)
# Handles BOTH: Login (email/pass) AND Medical Profile (age/gender)
class Patient(Base):
    __tablename__ = "patients"

    # Auth Fields (From Fastend User)
    patient_id = Column(Integer, primary_key=True, nullable=False)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))

    # Medical Fields (New additions)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    # You can update these fields later via a "Update Profile" endpoint

# 2. DOCTOR (Static Entity)
class Doctor(Base):
    __tablename__ = "doctors"
    doctor_id = Column(Integer, primary_key=True, nullable=False)
    name = Column(String, nullable=False) # e.g. "AI-Ophthalmologist"
    specialty = Column(String, nullable=False)

# 3. CONSULTATION (Replaces 'Post' from Fastend)
# This is the "Post" that belongs to a User (Patient)
class Consultation(Base):
    __tablename__ = "consultations"
    
    consultation_id = Column(Integer, primary_key=True, nullable=False)
    
    # RELATIONAL LINK: Matches 'owner_id' from Fastend
    patient_id = Column(Integer, ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False)
    
    status = Column(String, server_default="Active") # Triage, Active, Completed
    started_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))

    # Relationship to Patient
    patient = relationship("Patient") 

# 4. LAB ORDER (New Entity)
class LabOrder(Base):
    __tablename__ = "lab_orders"
    order_id = Column(Integer, primary_key=True, nullable=False)
    consultation_id = Column(Integer, ForeignKey("consultations.consultation_id", ondelete="CASCADE"), nullable=False)
    test_name = Column(String, nullable=False)
    status = Column(String, server_default="Pending")

# 5. LAB RESULT (New Entity)
class LabResult(Base):
    __tablename__ = "lab_results"
    result_id = Column(Integer, primary_key=True, nullable=False)
    order_id = Column(Integer, ForeignKey("lab_orders.order_id", ondelete="CASCADE"), nullable=False)
    findings = Column(Text, nullable=False)

# 6. MEDICAL REPORT (New Entity)
class MedicalReport(Base):
    __tablename__ = "medical_reports"
    report_id = Column(Integer, primary_key=True, nullable=False)
    consultation_id = Column(Integer, ForeignKey("consultations.consultation_id", ondelete="CASCADE"), nullable=False)
    diagnosis = Column(Text, nullable=False)
    treatment = Column(Text, nullable=False)