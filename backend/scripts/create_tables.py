from backend.db import engine, Base
from backend.models import team, driver, race, session, lap, stint, weather, telemetry

Base.metadata.create_all(bind=engine)
print("Tables created.")
