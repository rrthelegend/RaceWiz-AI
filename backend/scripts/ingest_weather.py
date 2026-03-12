import argparse

import fastf1
import pandas as pd

from app.db import SessionLocal
from app.models.race import Race
from app.models.session import Session
from app.models.weather import Weather


def ingest_weather(year: int) -> None:
    db = SessionLocal()
    try:
        sessions = (
            db.query(Session)
            .join(Session.race)
            .filter(Race.year == year)
            .all()
        )

        for s in sessions:
            try:
                session = fastf1.get_session(year, s.race.name, s.session_type)
                session.load(weather=True)

                weather_df = session.weather_data
                if weather_df is None or weather_df.empty:
                    continue

                rows: list[dict] = []
                for _, row in weather_df.iterrows():
                    t = row["Time"]
                    if pd.isna(t):
                        continue
                    time_offset = int(pd.to_timedelta(t).total_seconds())

                    rows.append(
                        {
                            "session_id": s.id,
                            "time_offset": time_offset,
                            "air_temp": row.get("AirTemp"),
                            "track_temp": row.get("TrackTemp"),
                            "humidity": row.get("Humidity"),
                            "pressure": row.get("Pressure"),
                            "rainfall": row.get("Rainfall"),
                            "wind_speed": row.get("WindSpeed"),
                            "wind_direction": row.get("WindDirection"),
                        }
                    )

                if rows:
                    db.bulk_insert_mappings(Weather, rows)
                    db.commit()
            except Exception:
                continue
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest weather data for a given year")
    parser.add_argument("--year", type=int, required=True, help="Championship year, e.g. 2023")
    args = parser.parse_args()
    ingest_weather(args.year)