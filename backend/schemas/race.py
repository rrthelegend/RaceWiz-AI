from datetime import date

from pydantic import BaseModel


class RaceBase(BaseModel):
    year: int
    round: int
    name: str
    location: str
    country: str
    date: date


class Race(RaceBase):
    id: int

    class Config:
        orm_mode = True

