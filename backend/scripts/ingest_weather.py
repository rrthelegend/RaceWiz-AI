import fastf1
import pandas as pd

from app.db import SessionLocal
from app.models.weather import Weather
from app.models.session import Session

YEAR = 2023


def ingest_weather():

    db = SessionLocal()

    sessions = db.query(Session).all()

    for s in sessions:

        try:

            session = fastf1.get_session(YEAR, s.race.name, s.session_type)
            session.load(weather=True)

            weather_df = session.weather_data

            weather_rows = []

            for _, row in weather_df.iterrows():

                weather_rows.append({
                    "session_id": s.id,
                    "time": row["Time"],
                    "air_temp": row["AirTemp"],
                    "track_temp": row["TrackTemp"],
                    "humidity": row["Humidity"],
                    "pressure": row["Pressure"],
                    "rainfall": row["Rainfall"],
                    "wind_speed": row["WindSpeed"]
                })

            db.bulk_insert_mappings(Weather, weather_rows)
            db.commit()

        except Exception:
            continue

    db.close()

    print("weather ingested")


if __name__ == "__main__":
    ingest_weather()