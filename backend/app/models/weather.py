from sqlalchemy import Column, Integer, Float, Boolean, ForeignKey
from app.db import Base

class Weather(Base):
    __tablename__ = "weather"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))

    time_offset = Column(Integer)  # seconds
    air_temp = Column(Float)
    track_temp = Column(Float)
    humidity = Column(Float)
    pressure = Column(Float)
    rainfall = Column(Boolean)
    wind_speed = Column(Float)
    wind_direction = Column(Integer)
