import fastf1
import math

fastf1.Cache.enable_cache("fastf1_cache")


def clean_records(records):
    """Convert NaN values to None so JSON works"""

    for row in records:
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                row[key] = None

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