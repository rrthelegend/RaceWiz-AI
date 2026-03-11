from collections.abc import Sequence

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.models.race import Race as RaceModel
from dependencies import get_db
from schemas.race import Race as RaceSchema


router = APIRouter(prefix="/races", tags=["races"])


@router.get("/", response_model=list[RaceSchema])
def list_races(
    year: int | None = Query(default=None, description="Filter by championship year"),
    db: Session = Depends(get_db),
) -> Sequence[RaceModel]:
    query = db.query(RaceModel)
    if year is not None:
        query = query.filter(RaceModel.year == year)
    return query.order_by(RaceModel.year.desc(), RaceModel.round).all()


@router.get("/{race_id}", response_model=RaceSchema)
def get_race(race_id: int, db: Session = Depends(get_db)) -> RaceModel:
    race = db.get(RaceModel, race_id)
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")
    return race

