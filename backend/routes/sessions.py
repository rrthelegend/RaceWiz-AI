from collections.abc import Sequence

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.session import Session as SessionModel
from schemas.session import Session as SessionSchema
from dependencies import get_db


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/race/{race_id}", response_model=list[SessionSchema])
def list_sessions_for_race(
    race_id: int,
    db: Session = Depends(get_db),
) -> Sequence[SessionModel]:
    return (
        db.query(SessionModel)
        .filter(SessionModel.race_id == race_id)
        .order_by(SessionModel.start_time)
        .all()
    )


@router.get("/{session_id}", response_model=SessionSchema)
def get_session(session_id: int, db: Session = Depends(get_db)) -> SessionModel:
    session = db.get(SessionModel, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

