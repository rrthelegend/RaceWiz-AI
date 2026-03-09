import fastf1
import pandas as pd

from app.db import SessionLocal
from app.models.lap import Lap
from app.models.session import Session

YEAR = 2023


def ingest_laps():

    db = SessionLocal()

    sessions = db.query(Session).filter(Session.session_type == "R").all()

    for s in sessions:

        print(f"Ingesting laps for {s.name}")

        session = fastf1.get_session(YEAR, s.race.name, "R")
        session.load()

        laps_df = session.laps

        lap_dicts = []

        for _, lap in laps_df.iterrows():

            if pd.isna(lap["LapTime"]):
                continue

            lap_time_ms = int(lap["LapTime"].total_seconds() * 1000)

            lap_dicts.append({
                "session_id": s.id,
                "driver_code": lap["Driver"],
                "lap_number": int(lap["LapNumber"]),
                "lap_time_ms": lap_time_ms,
                "stint": lap["Stint"],
                "compound": str(lap["Compound"]).upper(),
                "sector1": lap["Sector1Time"].total_seconds() if not pd.isna(lap["Sector1Time"]) else None,
                "sector2": lap["Sector2Time"].total_seconds() if not pd.isna(lap["Sector2Time"]) else None,
                "sector3": lap["Sector3Time"].total_seconds() if not pd.isna(lap["Sector3Time"]) else None
            })

        db.bulk_insert_mappings(Lap, lap_dicts)
        db.commit()

    db.close()

    print("laps ingested")


if __name__ == "__main__":
    ingest_laps()