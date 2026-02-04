from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship
from backend.db import Base

class Lap(Base):
    __tablename__ = "laps"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    driver_id = Column(Integer, ForeignKey("drivers.id"))

    lap_number = Column(Integer)
    lap_time_ms = Column(Float)
    compound = Column(String)
    tyre_life = Column(Integer)
    stint = Column(Integer)

    session = relationship("Session")
    driver = relationship("Driver")
