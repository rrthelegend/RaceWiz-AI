from pydantic import BaseModel


class TeamBase(BaseModel):
    name: str
    nationality: str | None = None


class Team(TeamBase):
    id: int

    class Config:
        orm_mode = True

