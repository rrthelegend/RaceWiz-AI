from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from dependencies import get_db
from services.analytics_service import AnalyticsService


router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/pace")
def analytics_pace(
    session_id: int = Query(..., description="Session identifier"),
    driver_id: int | None = Query(
        default=None, description="Optional driver identifier"
    ),
    db: Session = Depends(get_db),
):
    service = AnalyticsService(db)
    return service.lap_pace_delta(session_id=session_id, driver_id=driver_id)


@router.get("/tyres")
def analytics_tyres(
    session_id: int = Query(..., description="Session identifier"),
    driver_id: int | None = Query(
        default=None, description="Optional driver identifier"
    ),
    db: Session = Depends(get_db),
):
    service = AnalyticsService(db)
    return {
        "session_id": session_id,
        "driver_id": driver_id,
        "stints": service.tyre_stint_lengths(
            session_id=session_id, driver_id=driver_id
        ),
        "compound_summary": service.tyre_compound_summary(session_id=session_id),
        "undercut_overcut": service.undercut_overcut_summary(session_id=session_id),
    }


@router.get("/air-quality")
def analytics_air_quality(
    session_id: int = Query(..., description="Session identifier"),
    driver_id: int = Query(..., description="Driver identifier"),
    db: Session = Depends(get_db),
):
    service = AnalyticsService(db)
    return service.air_quality_indicators(session_id=session_id, driver_id=driver_id)

