import unittest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import date, datetime

from app.db import Base
from app.models.driver import Driver
from app.models.lap import Lap
from app.models.race import Race
from app.models.session import Session
from app.models.stint import Stint
from app.models.team import Team
from dependencies import get_db
from main import create_app


class ApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Use an isolated SQLite DB for tests (never hit Supabase).
        cls.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            future=True,
        )
        cls.TestingSessionLocal = sessionmaker(
            bind=cls.engine, autoflush=False, autocommit=False
        )
        Base.metadata.create_all(bind=cls.engine)

        app = create_app()

        def _override_get_db():
            db = cls.TestingSessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = _override_get_db
        cls.client = TestClient(app)

        db = cls.TestingSessionLocal()
        try:
            team = Team(name="Test Team", nationality="Test")
            db.add(team)
            db.flush()

            driver = Driver(code="TST", full_name="Test Driver", driver_number=99, team_id=team.id)
            db.add(driver)
            db.flush()

            race = Race(
                year=2023,
                round=1,
                name="Test GP",
                location="X",
                country="Y",
                date=date(2023, 3, 1),
            )
            db.add(race)
            db.flush()

            session = Session(
                race_id=race.id,
                session_type="R",
                start_time=datetime(2023, 3, 1, 0, 0, 0),
            )
            db.add(session)
            db.flush()

            # Minimal lap data across compounds
            laps = [
                Lap(session_id=session.id, driver_id=driver.id, lap_number=1, lap_time_ms=90000, compound="SOFT", tyre_life=1, stint=1),
                Lap(session_id=session.id, driver_id=driver.id, lap_number=2, lap_time_ms=90500, compound="SOFT", tyre_life=2, stint=1),
                Lap(session_id=session.id, driver_id=driver.id, lap_number=3, lap_time_ms=91000, compound="SOFT", tyre_life=3, stint=1),
                Lap(session_id=session.id, driver_id=driver.id, lap_number=4, lap_time_ms=92000, compound="MEDIUM", tyre_life=1, stint=2),
                Lap(session_id=session.id, driver_id=driver.id, lap_number=5, lap_time_ms=92500, compound="MEDIUM", tyre_life=2, stint=2),
            ]
            db.add_all(laps)

            stints = [
                Stint(session_id=session.id, driver_id=driver.id, compound="SOFT", start_lap=1, end_lap=3, stint_length=3),
                Stint(session_id=session.id, driver_id=driver.id, compound="MEDIUM", start_lap=4, end_lap=5, stint_length=2),
            ]
            db.add_all(stints)

            db.commit()

            cls.driver_id = driver.id
            cls.session_id = session.id
        finally:
            db.close()

    def test_health(self) -> None:
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_list_races(self) -> None:
        r = self.client.get("/races")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(isinstance(r.json(), list))

    def test_analytics_pace(self) -> None:
        r = self.client.get(f"/analytics/pace?session_id={self.session_id}&driver_id={self.driver_id}")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["session_id"], self.session_id)
        self.assertEqual(data["driver_id"], self.driver_id)
        self.assertIn("laps", data)

    def test_strategy_simulate(self) -> None:
        payload = {"driver": self.driver_id, "compound": "MEDIUM", "pit_lap": 3}
        r = self.client.post("/strategy/simulate", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["driver"], self.driver_id)
        self.assertEqual(data["compound"], "MEDIUM")
        self.assertEqual(data["pit_lap"], 3)
        self.assertIn("predicted_time_delta_ms", data)
        self.assertIn("risk_score", data)
        self.assertIn("strategy_viability_score", data)


if __name__ == "__main__":
    unittest.main()

