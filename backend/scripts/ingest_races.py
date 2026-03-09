import fastf1
from app.db import SessionLocal
from app.models.race import Race

YEAR = 2023


def ingest_races():

    schedule = fastf1.get_event_schedule(YEAR)

    db = SessionLocal()

    for _, row in schedule.iterrows():

        race = Race(
            year=YEAR,
            round=row["RoundNumber"],
            name=row["EventName"],
            location=row["Location"],
            country=row["Country"],
            date=row["EventDate"]
        )

        db.merge(race)

    db.commit()
    db.close()

    print("races ingested")


if __name__ == "__main__":
    ingest_races()