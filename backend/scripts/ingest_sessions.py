import fastf1

from app.db import SessionLocal
from app.models.race import Race
from app.models.session import Session

SESSION_TYPES = ["FP1", "FP2", "FP3", "Q", "R"]


def ingest_sessions() -> None:
    db = SessionLocal()
    try:
        races = db.query(Race).all()

        for race in races:
            for s in SESSION_TYPES:
                try:
                    session = fastf1.get_session(race.year, race.name, s)
                    session.load()

                    db_session = Session(
                        race_id=race.id,
                        session_type=s,
                        start_time=session.date,
                    )
                    db.merge(db_session)
                except Exception:
                    continue

        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    ingest_sessions()