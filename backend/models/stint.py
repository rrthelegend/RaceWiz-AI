from sqlalchemy import Column, Integer, String, ForeignKey
from backend.db import Base

class Stint(Base):
    __tablename__ = "stints"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    driver_id = Column(Integer, ForeignKey("drivers.id"))

    compound = Column(String)
    start_lap = Column(Integer)
    end_lap = Column(Integer)
    stint_length = Column(Integer)
