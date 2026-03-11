from pydantic import BaseModel


class WeatherBase(BaseModel):
    session_id: int
    time_offset: int
    air_temp: float | None = None
    track_temp: float | None = None
    humidity: float | None = None
    pressure: float | None = None
    rainfall: bool | None = None
    wind_speed: float | None = None
    wind_direction: int | None = None


class Weather(WeatherBase):
    id: int

    class Config:
        orm_mode = True

