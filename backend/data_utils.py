import fastf1
import pandas as pd
import numpy as np
import shutil
import os

CACHE_DIR = '/Users/rahulrajagopal/Documents/RaceWiz-AI/cache'
fastf1.Cache.enable_cache(CACHE_DIR)

def load_race_session(year: int, grand_prix: str, session_type: str = 'R'):
    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load()
        return session
    except Exception as e:
        print(f"Failed to load session: {e}")
        return None

def get_lap_data(session, driver: str = None):
    try:
        laps = session.laps.pick_driver(driver) if driver else session.laps.copy()
        laps = laps[['Driver', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 'TrackStatus', 'Stint', 'IsAccurate']]
        laps = laps[laps['IsAccurate'] == True].dropna(subset=['LapTime'])
        laps['LapTime(s)'] = laps['LapTime'].dt.total_seconds()
        laps.reset_index(drop=True, inplace=True)
        return laps
    except Exception as e:
        print(f"Failed to extract lap data: {e}")
        return pd.DataFrame()

def get_pit_stop_data(session):
    try:
        laps = session.laps
        pit_stops = laps[laps['PitInTime'].notnull() | laps['PitOutTime'].notnull()]
        pit_summary = pit_stops.groupby('Driver')['LapNumber'].count().reset_index()
        pit_summary.columns = ['Driver', 'NumberOfPitStops']
        return pit_summary
    except Exception as e:
        print(f"Failed to extract pit stop data: {e}")
        return pd.DataFrame()

def get_weather_data(session):
    try:
        weather = session.weather_data.copy()
        weather.reset_index(drop=True, inplace=True)
        return weather[['Time', 'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']]
    except Exception as e:
        print(f"Failed to get weather data: {e}")
        return pd.DataFrame()
