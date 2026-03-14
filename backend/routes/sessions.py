from fastapi import APIRouter, Query
from services.fastf1_service import get_sessions

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/")
def sessions(year: int = Query(...)):

    return get_sessions(year)