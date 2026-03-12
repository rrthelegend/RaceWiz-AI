from collections.abc import Sequence

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.selectors import get_session_by_natural_key
from app.models.driver import Driver as DriverModel
from app.models.stint import Stint as StintModel
from dependencies import get_db
from schemas.stint import Stint as StintSchema


router = APIRouter(prefix="/stints", tags=["stints"])


@router.get("/", response_model=list[StintSchema])
def list_stints(
    year: int = Query(..., description="Season year, e.g. 2023"),
    round: int = Query(..., description="Championship round number"),
    session: str = Query(..., description="Session code, e.g. FP1, FP2, FP3, Q, SQ, R"),
    driver: str | None = Query(
        default=None, description="Optional driver code filter, e.g. VER, HAM"
    ),
    compound: str | None = Query(
        default=None, description="Optional tyre compound filter, e.g. SOFT"
    ),
    db: Session = Depends(get_db),
) -> Sequence[StintModel]:
    db_session = get_session_by_natural_key(db, year=year, round=round, session_code=session)

    query = db.query(StintModel).filter(StintModel.session_id == db_session.id)

    if driver is not None:
        query = query.join(DriverModel).filter(DriverModel.code == driver.upper())

    if compound is not None:
        query = query.filter(StintModel.compound == compound.upper())

    return query.order_by(StintModel.start_lap).all()
