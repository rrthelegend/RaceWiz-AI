from collections.abc import Sequence

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.team import Team as TeamModel
from dependencies import get_db
from schemas.team import Team as TeamSchema


router = APIRouter(prefix="/teams", tags=["teams"])


@router.get("/", response_model=list[TeamSchema])
def list_teams(db: Session = Depends(get_db)) -> Sequence[TeamModel]:
    return db.query(TeamModel).order_by(TeamModel.name).all()


@router.get("/{team_id}", response_model=TeamSchema)
def get_team(team_id: int, db: Session = Depends(get_db)) -> TeamModel:
    team = db.get(TeamModel, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team

