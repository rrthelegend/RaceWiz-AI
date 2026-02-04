from sqlalchemy import Column, Integer, String, Date
from backend.db import Base

class Race(Base):
    __tablename__ = "races"

    id = Column(Integer, primary_key=True)
    year = Column(Integer, index=True)
    round = Column(Integer)
    name = Column(String)
    location = Column(String)
    country = Column(String)
    date = Column(Date)
