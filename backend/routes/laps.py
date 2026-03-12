from collections.abc import Sequence

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.selectors import get_session_by_natural_key
from app.models.driver import Driver as DriverModel
from app.models.lap import Lap as LapModel
from dependencies import get_db
from schemas.lap import Lap as LapSchema


router = APIRouter(prefix="/laps", tags=["laps"])


@router.get("/", response_model=list[LapSchema])
def list_laps(
    year: int = Query(..., description="Season year, e.g. 2023"),
    round: int = Query(..., description="Championship round number"),
    session: str = Query(..., description="Session code, e.g. FP1, FP2, FP3, Q, SQ, R"),
    driver: str | None = Query(
        default=None, description="Optional driver code filter, e.g. VER, HAM"
    ),
    stint: int | None = Query(default=None, description="Optional stint number filter"),
    compound: str | None = Query(
        default=None, description="Optional tyre compound filter, e.g. SOFT"
    ),
    db: Session = Depends(get_db),
) -> Sequence[LapModel]:
    db_session = get_session_by_natural_key(db, year=year, round=round, session_code=session)

    query = (
        db.query(LapModel)
        .filter(LapModel.session_id == db_session.id)
    )

    if driver is not None:
        query = query.join(DriverModel).filter(DriverModel.code == driver.upper())

    if stint is not None:
        query = query.filter(LapModel.stint == stint)

    if compound is not None:
        query = query.filter(LapModel.compound == compound.upper())

    return query.order_by(LapModel.lap_number).all()
