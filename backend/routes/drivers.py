from collections.abc import Sequence

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.selectors import get_session_by_natural_key
from app.models.driver import Driver as DriverModel
from app.models.lap import Lap as LapModel
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
    db_session = get_session_by_natural_key(db, year=year, round=round, session_code=session)

    query = (
        db.query(DriverModel)
        .join(LapModel, LapModel.driver_id == DriverModel.id)
        .filter(LapModel.session_id == db_session.id)
    )

    if driver is not None:
        query = query.filter(DriverModel.code == driver.upper())

    return query.distinct().order_by(DriverModel.code).all()
