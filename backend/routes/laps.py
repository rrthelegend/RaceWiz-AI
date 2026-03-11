from collections.abc import Sequence

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.models.lap import Lap as LapModel
from dependencies import get_db
from schemas.lap import Lap as LapSchema


router = APIRouter(prefix="/laps", tags=["laps"])


@router.get("/", response_model=list[LapSchema])
def list_laps(
    session_id: int = Query(..., description="Filter by session"),
    driver_id: int | None = Query(default=None, description="Optional driver filter"),
    db: Session = Depends(get_db),
) -> Sequence[LapModel]:
    query = db.query(LapModel).filter(LapModel.session_id == session_id)
    if driver_id is not None:
        query = query.filter(LapModel.driver_id == driver_id)
    return query.order_by(LapModel.lap_number).all()


@router.get("/{lap_id}", response_model=LapSchema)
def get_lap(lap_id: int, db: Session = Depends(get_db)) -> LapModel:
    lap = db.get(LapModel, lap_id)
    if not lap:
        raise HTTPException(status_code=404, detail="Lap not found")
    return lap

