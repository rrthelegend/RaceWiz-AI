from sqlalchemy import Column, Integer, Float, ForeignKey
from app.db import Base

class Telemetry(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True)
    lap_id = Column(Integer, ForeignKey("laps.id"))

    distance = Column(Float)
    speed = Column(Float)
    throttle = Column(Float)
    brake = Column(Float)
    gear = Column(Integer)
    rpm = Column(Integer)
