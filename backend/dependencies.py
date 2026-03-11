from collections.abc import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session and
    makes sure it is closed after the request finishes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DbSessionDep = Depends(get_db)

