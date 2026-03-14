from fastapi import APIRouter, Query
from services.analytics_service import pace_analysis
from services.analytics_service import tyre_strategy
from services.analytics_service import degradation

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/pace")
def pace(
    year: int = Query(...),
    round: int = Query(...),
    session: str = Query(...)
):
    return pace_analysis(year, round, session)

@router.get("/strategy")
def strategy(
    year: int = Query(...),
    round: int = Query(...),
    session: str = Query(...)
):
    return tyre_strategy(year, round, session)


@router.get("/degradation")
def degradation_endpoint(
    year: int = Query(...),
    round: int = Query(...),
    session: str = Query(...),
    driver: str = Query(...)
):
    return degradation(year, round, session, driver)