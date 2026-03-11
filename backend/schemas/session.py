from datetime import datetime

from pydantic import BaseModel


class SessionBase(BaseModel):
    race_id: int
    session_type: str
    start_time: datetime


class Session(SessionBase):
    id: int

    class Config:
        orm_mode = True

