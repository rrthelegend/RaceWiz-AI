import argparse

import fastf1
import pandas as pd

from app.db import SessionLocal
from app.models.lap import Lap
from app.models.race import Race
from app.models.session import Session


def ingest_laps(year: int) -> None:
    db = SessionLocal()
    try:
        sessions = (
            db.query(Session)
            .join(Session.race)
            .filter(Session.session_type == "R", Race.year == year)
            .all()
        )

        for s in sessions:
            print(f"Ingesting laps for {year} {s.race.name}")

            session = fastf1.get_session(year, s.race.name, "R")
            session.load()

            laps_df = session.laps
            lap_dicts: list[dict] = []

            for _, lap in laps_df.iterrows():
                if pd.isna(lap["LapTime"]):
                    continue

                lap_time_ms = int(lap["LapTime"].total_seconds() * 1000)

                lap_dicts.append(
                    {
                        "session_id": s.id,
                        "driver_id": None,
                        "lap_number": int(lap["LapNumber"]),
                        "lap_time_ms": lap_time_ms,
                        "stint": lap.get("Stint"),
                        "compound": str(lap.get("Compound") or "").upper() or None,
                        "tyre_life": lap.get("TyreLife"),
                    }
                )

            if lap_dicts:
                db.bulk_insert_mappings(Lap, lap_dicts)
                db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest lap data for a given year")
    parser.add_argument("--year", type=int, required=True, help="Championship year, e.g. 2023")
    args = parser.parse_args()
    ingest_laps(args.year)