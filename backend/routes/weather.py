from collections.abc import Sequence

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.selectors import get_session_by_natural_key
from app.models.weather import Weather as WeatherModel
from dependencies import get_db
from schemas.weather import Weather as WeatherSchema


router = APIRouter(prefix="/weather", tags=["weather"])


@router.get("/", response_model=list[WeatherSchema])
def list_weather(
    year: int = Query(..., description="Season year, e.g. 2023"),
    round: int = Query(..., description="Championship round number"),
    session: str = Query(..., description="Session code, e.g. FP1, FP2, FP3, Q, SQ, R"),
    db: Session = Depends(get_db),
) -> Sequence[WeatherModel]:
    db_session = get_session_by_natural_key(db, year=year, round=round, session_code=session)

    return (
        db.query(WeatherModel)
        .filter(WeatherModel.session_id == db_session.id)
        .order_by(WeatherModel.time_offset)
        .all()
    )
