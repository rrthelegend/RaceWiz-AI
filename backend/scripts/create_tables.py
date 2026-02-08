from app.db import engine, Base
from app.models import team, driver, race, session, lap, stint, weather, telemetry

Base.metadata.create_all(bind=engine)

print("Tables created.")
