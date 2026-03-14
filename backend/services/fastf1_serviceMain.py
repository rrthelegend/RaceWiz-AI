import fastf1 
import pandas as pd

class FastF1Service:
    def __init__(self, cache_dir="fastf1_cache"):
        fastf1.Cache.enable_cache(cache_dir)

    def load_session(self, year: int, event: str, session_type: str):
        try:
            session = fastf1.get_session(year, event, session_type)
            session.load(weather=True)
            return session
        except Exception as e:
            raise RuntimeError(f"Failed to load session : {year} {event} {session_type}") from e
        
    def get_laps(self, session) -> pd.DataFrame:
        if session.laps is None or session.laps.empty:
            raise ValueError("Session has no lap data loaded")

        laps = session.laps.copy()
        laps = laps.dropna(subset=["LapTime"])
        laps["LapTimeMs"] = laps["LapTime"].dt.total_seconds() * 1000

        if "Compound" in laps.columns:
            laps["Compound"] = laps["Compound"].str.upper()

        laps["LapNumber"] = laps["LapNumber"].astype(int)
        
        laps = laps.sort_values(
            by=["Driver", "LapNumber"], ascending=[True, True]
        ).reset_index(drop=True)

        return laps
    
    def get_weather(self, session) -> pd.DataFrame:
        weather = session.weather_data
        if weather is None or weather.empty:
            raise ValueError("Weather data not available for this session")

        weather = weather.copy()

        if "Time" in weather.columns:
            weather["Time"] = pd.to_timedelta(weather["Time"])

        stable_columns = [
            "Time",
            "AirTemp",
            "TrackTemp",
            "Humidity",
            "Pressure",
            "Rainfall",
            "WindDirection",
            "WindSpeed",
        ]

        available = [c for c in stable_columns if c in weather.columns]
        weather = weather[available]

        weather = weather.reset_index(drop=True)
        return weather

    
    def get_results(self, session) -> pd.DataFrame:
        if session.results is None or session.results.empty:
            raise ValueError("Session has no result data")

        results = session.results.copy()

        
        keep_cols = [
            "Position",
            "DriverNumber",
            "Abbreviation",
            "FullName",
            "TeamName",
            "Time",
            "Status",
        ]

        available_cols = [c for c in keep_cols if c in results.columns]
        results = results[available_cols]

        return results.reset_index(drop=True)

    
    def get_fastest_lap_telemetry(
        self, session, driver_code: str
    ) -> pd.DataFrame:
        laps = session.laps.pick_driver(driver_code)

        if laps.empty:
            raise ValueError(f"No laps found for driver {driver_code}")

        fastest_lap = laps.pick_fastest()

        if fastest_lap is None:
            raise ValueError(f"No fastest lap found for driver {driver_code}")

        telemetry = (
            fastest_lap.get_telemetry()
            .add_distance()
            .reset_index(drop=True)
        )

        return telemetry

    
    def get_stints(self, session) -> pd.DataFrame:
        laps = self.get_laps(session)

        stints = (
            laps.groupby(["Driver", "Stint", "Compound"])
            .agg(
                StartLap=("LapNumber", "min"),
                EndLap=("LapNumber", "max"),
                StintLength=("LapNumber", "count"),
            )
            .reset_index()
        )

        return stints
    
    