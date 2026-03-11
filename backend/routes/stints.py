from collections.abc import Sequence

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.models.stint import Stint as StintModel
from dependencies import get_db
from schemas.stint import Stint as StintSchema


router = APIRouter(prefix="/stints", tags=["stints"])


@router.get("/", response_model=list[StintSchema])
def list_stints(
    session_id: int = Query(..., description="Filter by session"),
    driver_id: int | None = Query(default=None, description="Optional driver filter"),
    db: Session = Depends(get_db),
) -> Sequence[StintModel]:
    query = db.query(StintModel).filter(StintModel.session_id == session_id)
    if driver_id is not None:
        query = query.filter(StintModel.driver_id == driver_id)
    return query.order_by(StintModel.start_lap).all()


@router.get("/{stint_id}", response_model=StintSchema)
def get_stint(stint_id: int, db: Session = Depends(get_db)) -> StintModel:
    stint = db.get(StintModel, stint_id)
    if not stint:
        raise HTTPException(status_code=404, detail="Stint not found")
    return stint

