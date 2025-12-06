from fastapi import FastAPI
from .api import router
from .cors_config import add_cors_middleware
from .routers import users, oauth
from sqlalchemy.orm import Session
from . import models, database

# Define your static doctors
SPECIALISTS = [
    {"name": "Dr. A. Eye", "specialty": "Ophthalmologist"},
    {"name": "Dr. B. Skin", "specialty": "Dermatologist"},
    {"name": "Dr. C. Bone", "specialty": "Orthopedist"},
    {"name": "Dr. D. Child", "specialty": "Pediatrician"},
    {"name": "Dr. E. Throat", "specialty": "ENT"},
    {"name": "Dr. F. Wom", "specialty": "Gynecologist"},
    {"name": "Dr. G. Mind", "specialty": "Psychiatrist"},
    {"name": "Dr. H. Gen", "specialty": "Internal Medicine"},
    {"name": "Dr. I. General", "specialty": "GP"},
]

# Create a startup function
def seed_doctors():
    db = database.SessionLocal()
    try:
        # Check if empty
        if db.query(models.Doctor).count() == 0:
            print("🌱 Seeding Doctors Table...")
            for doc in SPECIALISTS:
                new_doc = models.Doctor(name=doc["name"], specialty=doc["specialty"])
                db.add(new_doc)
            db.commit()
            print("✅ Doctors Seeded.")
    finally:
        db.close()

# Run it on startup
models.Base.metadata.create_all(bind=database.engine)
seed_doctors() # <--- Add this call


app = FastAPI()
add_cors_middleware(app)
app.include_router(users.router)
app.include_router(oauth.router)
app.include_router(router)