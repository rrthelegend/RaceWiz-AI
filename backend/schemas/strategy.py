from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, conint


TyreCompound = Literal["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


class StrategySimulateRequest(BaseModel):
    driver: conint(gt=0) = Field(..., description="Driver id")
    compound: TyreCompound = Field(..., description="Target tyre compound")
    pit_lap: conint(gt=0) = Field(..., description="Lap number of the pit stop")


class StrategySimulateDetails(BaseModel):
    total_laps: int
    laps_after_pit: int
    baseline_post_pit_avg_lap_ms: float
    simulated_post_pit_avg_lap_ms: float


class StrategySimulateResponse(BaseModel):
    session_id: int | None = Field(
        default=None, description="Resolved session id used for simulation"
    )
    driver: int
    compound: TyreCompound
    pit_lap: int

    predicted_time_delta_ms: float = Field(
        ..., description="Positive = predicted time gained, negative = lost"
    )
    risk_score: float = Field(..., ge=0.0, le=1.0)
    strategy_viability_score: float = Field(..., ge=0.0, le=100.0)
    details: StrategySimulateDetails | dict

