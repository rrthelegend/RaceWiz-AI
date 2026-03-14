from fastapi import APIRouter, Query
from services.analytics_service import pace_analysis

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/pace")
def pace(
    year: int = Query(...),
    round: int = Query(...),
    session: str = Query(...)
):

    return pace_analysis(year, round, session)