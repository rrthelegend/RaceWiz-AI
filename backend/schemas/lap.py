from pydantic import BaseModel


class LapBase(BaseModel):
    session_id: int
    driver_id: int
    lap_number: int
    lap_time_ms: float
    compound: str | None = None
    tyre_life: int | None = None
    stint: int | None = None


class Lap(LapBase):
    id: int

    class Config:
        orm_mode = True

