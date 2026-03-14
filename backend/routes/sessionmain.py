from collections.abc import Sequence

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from app.models.race import Race
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

    race = (
        db.query(Race)
        .filter(Race.year == year)
        .filter(Race.round == round)
        .first()
    )

    if race is None:
        raise HTTPException(status_code=404, detail="Race not found")

    sessions = (
        db.query(SessionModel)
        .filter(SessionModel.race_id == race.id)
        .order_by(SessionModel.start_time)
        .all()
    )

    return sessions