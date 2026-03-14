import pandas as pd
import fastf1

fastf1.Cache.enable_cache(cache_dir="fastf1_cache")


def pace_analysis(year, round, session_type):

    session = fastf1.get_session(year, round, session_type)
    session.load()

    laps = session.laps.pick_quicklaps()

    pace = laps.groupby("Driver")["LapTime"].mean()

    return pace.sort_values().to_dict()