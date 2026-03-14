from fastapi import APIRouter, Query
from services.fastf1_service import get_laps

router = APIRouter(prefix="/laps", tags=["laps"])


@router.get("/")
def laps(
    year: int = Query(...),
    round: int = Query(...),
    session: str = Query(...)
):

    return get_laps(year, round, session)