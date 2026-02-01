from sqlalchemy import Column, Integer, String, ForeignKey
from app.db import Base

class Driver(Base):
    __tablename__ = "drivers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    code = Column(String, unique=True)
    team_id = Column(Integer, ForeignKey("teams.id"))

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.models.driver import Driver
from app.schemas.driver import DriverSchema

router = APIRouter(prefix="/drivers", tags=["drivers"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/", response_model=list[DriverSchema])
def get_drivers(db: Session = Depends(get_db)):
    return db.query(Driver).all()
