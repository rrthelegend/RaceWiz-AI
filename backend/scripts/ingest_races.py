import argparse

import fastf1

from app.db import SessionLocal
from app.models.race import Race


def ingest_races(year: int) -> None:
    schedule = fastf1.get_event_schedule(year)

    db = SessionLocal()
    try:
        for _, row in schedule.iterrows():
            race = Race(
                year=year,
                round=row["RoundNumber"],
                name=row["EventName"],
                location=row["Location"],
                country=row["Country"],
                date=row["EventDate"],
            )
            db.merge(race)
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest F1 races for a given year")
    parser.add_argument("--year", type=int, required=True, help="Championship year, e.g. 2023")
    args = parser.parse_args()
    ingest_races(args.year)