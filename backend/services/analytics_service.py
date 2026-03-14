import fastf1
import pandas as pd

fastf1.Cache.enable_cache("fastf1_cache")


def pace_analysis(year, round, session_type):

    session = fastf1.get_session(year, round, session_type)
    session.load()

    laps = session.laps.pick_quicklaps()

    pace = laps.groupby("Driver")["LapTime"].mean()

    pace = pace.sort_values()

    result = {}

    for driver, lap_time in pace.items():
        seconds = lap_time.total_seconds()
        minutes = int(seconds // 60)
        seconds = seconds % 60

        result[driver] = f"{minutes}:{seconds:06.3f}"

    return result


def tyre_strategy(year, round, session_type):

    session = fastf1.get_session(year, round, session_type)
    session.load()

    laps = session.laps

    strategies = {}

    for driver in laps["Driver"].unique():

        driver_laps = laps.pick_driver(driver)

        stints = driver_laps.groupby("Stint")["Compound"].first()

        strategies[driver] = list(stints.values)

    return strategies

def degradation(year, round, session_type, driver):

    session = fastf1.get_session(year, round, session_type)
    session.load()

    laps = session.laps.pick_driver(driver)

    result = []

    for _, lap in laps.iterrows():

        if pd.isna(lap["LapTime"]):
            continue

        result.append({
            "lap": int(lap["LapNumber"]),
            "time": lap["LapTime"].total_seconds()
        })

    return result