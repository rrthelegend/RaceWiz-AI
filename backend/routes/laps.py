from collections.abc import Sequence

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from app.models.driver import Driver as DriverModel
from app.models.lap import Lap as LapModel
from app.models.race import Race
from app.models.session import Session as SessionModel
from dependencies import get_db
from schemas.lap import Lap as LapSchema

router = APIRouter(prefix="/laps", tags=["laps"])


@router.get("/", response_model=list[LapSchema])
def list_laps(
    year: int = Query(...),
    round: int = Query(...),
    session: str = Query(...),
    driver: str | None = Query(default=None),
    stint: int | None = Query(default=None),
    compound: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> Sequence[LapModel]:

    # normalize session codes
    session = session.upper()

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

    query = db.query(LapModel).filter(LapModel.session_id == db_session.id)

    if driver is not None:
        query = query.join(DriverModel).filter(DriverModel.code == driver.upper())

    if stint is not None:
        query = query.filter(LapModel.stint == stint)

    if compound is not None:
        query = query.filter(LapModel.compound == compound.upper())

    laps = query.order_by(LapModel.lap_number).all()

    return laps