import fastf1
import math
import pandas as pd

fastf1.Cache.enable_cache("fastf1_cache")


def format_timedelta(value):
    """Convert pandas Timedelta to F1 style time"""

    if pd.isna(value):
        return None

    total_seconds = value.total_seconds()

    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)

    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


def clean_records(records):

    for row in records:
        for key, value in row.items():

            # Convert NaN
            if isinstance(value, float) and math.isnan(value):
                row[key] = None

            # Convert pandas Timedelta (lap times)
            elif isinstance(value, pd.Timedelta):
                row[key] = format_timedelta(value)

            # Convert Timestamp
            elif isinstance(value, pd.Timestamp):
                row[key] = value.isoformat()

    return records


def load_session(year: int, round: int, session_type: str):

    if session_type.upper() == "R":
        session_type = "Race"

    session = fastf1.get_session(year, round, session_type)
    session.load()

    return session


def get_sessions(year: int):

    schedule = fastf1.get_event_schedule(year)

    records = schedule.to_dict(orient="records")

    return clean_records(records)


def get_laps(year: int, round: int, session_type: str):

    session = load_session(year, round, session_type)

    laps = session.laps

    records = laps.to_dict(orient="records")

    return clean_records(records)


def get_drivers(year: int, round: int, session_type: str):

    session = load_session(year, round, session_type)

    results = session.results

    records = results.to_dict(orient="records")

    return clean_records(records)