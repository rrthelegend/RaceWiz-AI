import fastf1
from app.db import SessionLocal
from app.models.driver import Driver
from app.models.team import Team
from app.models.session import Session

YEAR = 2023


def ingest_drivers_teams():

    db = SessionLocal()

    sessions = db.query(Session).filter(Session.session_type == "R").all()

    for s in sessions:

        session = fastf1.get_session(YEAR, s.race.name, "R")
        session.load()

        results = session.results

        for _, row in results.iterrows():

            team = Team(name=row["TeamName"])
            db.merge(team)

            driver = Driver(
                number=row["DriverNumber"],
                code=row["Abbreviation"],
                name=row["FullName"],
                team=row["TeamName"]
            )

            db.merge(driver)

    db.commit()
    db.close()

    print("drivers & teams ingested")


if __name__ == "__main__":
    ingest_drivers_teams()