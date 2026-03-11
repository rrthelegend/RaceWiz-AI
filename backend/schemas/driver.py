from pydantic import BaseModel


class DriverBase(BaseModel):
    code: str
    full_name: str
    driver_number: int
    team_id: int | None = None


class Driver(DriverBase):
    id: int

    class Config:
        orm_mode = True

