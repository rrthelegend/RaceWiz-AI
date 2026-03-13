from collections.abc import Sequence

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from app.models.driver import Driver as DriverModel
from app.models.lap import Lap as LapModel
from app.models.race import Race
from app.models.session import Session as SessionModel
from dependencies import get_db
from schemas.driver import Driver as DriverSchema


router = APIRouter(prefix="/drivers", tags=["drivers"])


@router.get("/", response_model=list[DriverSchema])
def list_drivers(
    year: int = Query(..., description="Season year, e.g. 2023"),
    round: int = Query(..., description="Championship round number"),
    session: str = Query(..., description="Session code, e.g. FP1, FP2, FP3, Q, SQ, R"),
    driver: str | None = Query(
        default=None, description="Optional driver code filter, e.g. VER, HAM"
    ),
    db: Session = Depends(get_db),
) -> Sequence[DriverModel]:

    session = session.upper()

    # normalize race session codes
    if session == "R":
        session = "Race"

    race = (
        db.query(Race)
        .filter(Race.year == year)
        .filter(Race.round == round)
        .first()
    )

    if race is None:
        raise HTTPException(status_code=404, detail="Race not found")

    db_session = (
        db.query(SessionModel)
        .filter(SessionModel.race_id == race.id)
        .filter(SessionModel.session_type == session)
        .first()
    )

    if db_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    query = (
        db.query(DriverModel)
        .join(LapModel, LapModel.driver_id == DriverModel.id)
        .filter(LapModel.session_id == db_session.id)
    )

    if driver is not None:
        query = query.filter(DriverModel.code == driver.upper())

    drivers = query.distinct().order_by(DriverModel.code).all()

    return drivers