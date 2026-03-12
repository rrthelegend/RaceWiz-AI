from collections.abc import Sequence

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.selectors import get_race_by_year_round
from app.models.session import Session as SessionModel
from dependencies import get_db
from schemas.session import Session as SessionSchema


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/", response_model=list[SessionSchema])
def list_sessions(
    year: int = Query(..., description="Season year, e.g. 2023"),
    round: int = Query(..., description="Championship round number"),
    db: Session = Depends(get_db),
) -> Sequence[SessionModel]:
    race = get_race_by_year_round(db, year=year, round=round)
    return (
        db.query(SessionModel)
        .filter(SessionModel.race_id == race.id)
        .order_by(SessionModel.start_time)
        .all()
    )
