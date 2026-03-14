from fastapi import APIRouter, Query
from services.fastf1_service import get_drivers

router = APIRouter(prefix="/drivers", tags=["drivers"])


@router.get("/")
def drivers(
    year: int = Query(...),
    round: int = Query(...),
    session: str = Query(...)
):

    return get_drivers(year, round, session)