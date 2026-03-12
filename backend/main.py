from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError
from starlette.requests import Request

from app.db import Base, engine, DATABASE_URL
from routes import (
    analytics,
    drivers,
    health,
    laps,
    races,
    sessions,
    stints,
    strategy,
    teams,
    weather,
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="RaceWiz AI Backend",
        description="Formula 1 race strategy analysis and simulation API",
        version="0.1.0",
    )

    app.include_router(races.router)
    app.include_router(sessions.router)
    app.include_router(drivers.router)
    app.include_router(teams.router)
    app.include_router(laps.router)
    app.include_router(stints.router)
    app.include_router(weather.router)
    app.include_router(analytics.router)
    app.include_router(strategy.router)
    app.include_router(health.router)

    @app.get("/")
    async def root():
        return {"message": "RaceWiz AI backend is running"}

    @app.on_event("startup")
    def _startup() -> None:
        if DATABASE_URL.startswith("sqlite"):
            Base.metadata.create_all(bind=engine)

    @app.exception_handler(OperationalError)
    async def db_operational_error_handler(
        request: Request, exc: OperationalError
    ) -> JSONResponse:
        """
        Normalize database connectivity/runtime errors into a clean 503
        response instead of an unhandled 500 with a long traceback.
        """
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Database unavailable or connection error",
                "error": str(exc.orig) if getattr(exc, "orig", None) else str(exc),
            },
        )

    return app


app = create_app()

