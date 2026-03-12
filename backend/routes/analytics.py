from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.selectors import get_driver_by_code, get_session_by_natural_key
from dependencies import get_db
from services.analytics_service import AnalyticsService


router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/pace")
def analytics_pace(
    year: int = Query(..., description="Season year, e.g. 2023"),
    round: int = Query(..., description="Championship round number"),
    session: str = Query(..., description="Session code, e.g. FP1, FP2, FP3, Q, SQ, R"),
    driver: str | None = Query(
        default=None, description="Optional driver code filter, e.g. VER, HAM"
    ),
    db: Session = Depends(get_db),
):
    db_session = get_session_by_natural_key(db, year=year, round=round, session_code=session)
    driver_id: int | None = None
    if driver is not None:
        driver_id = get_driver_by_code(db, driver).id

    service = AnalyticsService(db)
    return service.lap_pace_delta(session_id=db_session.id, driver_id=driver_id)


@router.get("/tyres")
def analytics_tyres(
    year: int = Query(..., description="Season year, e.g. 2023"),
    round: int = Query(..., description="Championship round number"),
    session: str = Query(..., description="Session code, e.g. FP1, FP2, FP3, Q, SQ, R"),
    driver: str | None = Query(
        default=None, description="Optional driver code filter, e.g. VER, HAM"
    ),
    db: Session = Depends(get_db),
):
    db_session = get_session_by_natural_key(db, year=year, round=round, session_code=session)
    driver_id: int | None = None
    if driver is not None:
        driver_id = get_driver_by_code(db, driver).id

    service = AnalyticsService(db)
    return {
        "session_id": db_session.id,
        "driver_id": driver_id,
        "stints": service.tyre_stint_lengths(
            session_id=db_session.id, driver_id=driver_id
        ),
        "compound_summary": service.tyre_compound_summary(session_id=db_session.id),
        "undercut_overcut": service.undercut_overcut_summary(session_id=db_session.id),
    }


@router.get("/air-quality")
def analytics_air_quality(
    year: int = Query(..., description="Season year, e.g. 2023"),
    round: int = Query(..., description="Championship round number"),
    session: str = Query(..., description="Session code, e.g. FP1, FP2, FP3, Q, SQ, R"),
    driver: str = Query(..., description="Driver code, e.g. VER, HAM"),
    db: Session = Depends(get_db),
):
    db_session = get_session_by_natural_key(db, year=year, round=round, session_code=session)
    driver_id = get_driver_by_code(db, driver).id

    service = AnalyticsService(db)
    return service.air_quality_indicators(session_id=db_session.id, driver_id=driver_id)

