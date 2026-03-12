import argparse

import fastf1

from app.db import SessionLocal
from app.models.driver import Driver
from app.models.race import Race
from app.models.session import Session
from app.models.team import Team


def ingest_drivers_teams(year: int) -> None:
    db = SessionLocal()
    try:
        sessions = (
            db.query(Session)
            .join(Session.race)
            .filter(Session.session_type == "R", Race.year == year)
            .all()
        )

        for s in sessions:
            session = fastf1.get_session(year, s.race.name, "R")
            session.load()

            results = session.results

            for _, row in results.iterrows():
                team_name = row["TeamName"]
                team = db.query(Team).filter(Team.name == team_name).one_or_none()
                if not team:
                    team = Team(name=team_name, nationality=None)
                    db.add(team)
                    db.flush()

                driver_number = int(row["DriverNumber"])
                driver = (
                    db.query(Driver)
                    .filter(Driver.driver_number == driver_number)
                    .one_or_none()
                )
                if not driver:
                    driver = Driver(
                        driver_number=driver_number,
                        code=row["Abbreviation"],
                        full_name=row["FullName"],
                        team_id=team.id,
                    )
                    db.add(driver)
                else:
                    driver.code = row["Abbreviation"]
                    driver.full_name = row["FullName"]
                    driver.team_id = team.id

        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest drivers and teams for a given year")
    parser.add_argument("--year", type=int, required=True, help="Championship year, e.g. 2023")
    args = parser.parse_args()
    ingest_drivers_teams(args.year)