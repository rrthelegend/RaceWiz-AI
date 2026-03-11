from pydantic import BaseModel


class StintBase(BaseModel):
    session_id: int
    driver_id: int
    compound: str
    start_lap: int
    end_lap: int
    stint_length: int


class Stint(StintBase):
    id: int

    class Config:
        orm_mode = True

