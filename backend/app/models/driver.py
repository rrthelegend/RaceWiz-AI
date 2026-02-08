from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from app.db import Base

class Driver(Base):
    __tablename__ = "drivers"

    id = Column(Integer, primary_key=True)
    code = Column(String(3), index=True)
    full_name = Column(String)
    driver_number = Column(Integer)

    team_id = Column(Integer, ForeignKey("teams.id"))
    team = relationship("Team")
