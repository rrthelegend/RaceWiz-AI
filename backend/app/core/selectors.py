from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.driver import Driver
from app.models.race import Race
from app.models.session import Session as SessionModel


def get_race_by_year_round(db: Session, year: int, round: int) -> Race:
    race = (
        db.query(Race)
        .filter(Race.year == year, Race.round == round)
        .one_or_none()
    )
    if not race:
        raise HTTPException(
            status_code=404,
            detail=f"Race not found for year={year}, round={round}",
        )
    return race


def get_session_by_natural_key(
    db: Session, year: int, round: int, session_code: str
) -> SessionModel:
    session_code = session_code.upper()
    race = get_race_by_year_round(db, year, round)
    session = (
        db.query(SessionModel)
        .filter(
            SessionModel.race_id == race.id,
            SessionModel.session_type == session_code,
        )
        .one_or_none()
    )
    if not session:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session not found for year={year}, round={round}, "
                f"session='{session_code}'"
            ),
        )
    return session


def get_driver_by_code(db: Session, driver_code: str) -> Driver:
    code = driver_code.upper()
    driver = db.query(Driver).filter(Driver.code == code).one_or_none()
    if not driver:
        raise HTTPException(
            status_code=404,
            detail=f"Driver not found for code='{code}'",
        )
    return driver

