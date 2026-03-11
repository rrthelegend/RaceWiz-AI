from collections.abc import Sequence

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.models.weather import Weather as WeatherModel
from dependencies import get_db
from schemas.weather import Weather as WeatherSchema


router = APIRouter(prefix="/weather", tags=["weather"])


@router.get("/", response_model=list[WeatherSchema])
def list_weather(
    session_id: int = Query(..., description="Session identifier"),
    db: Session = Depends(get_db),
) -> Sequence[WeatherModel]:
    return (
        db.query(WeatherModel)
        .filter(WeatherModel.session_id == session_id)
        .order_by(WeatherModel.time_offset)
        .all()
    )


@router.get("/{weather_id}", response_model=WeatherSchema)
def get_weather(weather_id: int, db: Session = Depends(get_db)) -> WeatherModel:
    record = db.get(WeatherModel, weather_id)
    if not record:
        raise HTTPException(status_code=404, detail="Weather record not found")
    return record

