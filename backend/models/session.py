from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from backend.db import Base

class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id"))
    session_type = Column(String)
    start_time = Column(DateTime)

    race = relationship("Race")
