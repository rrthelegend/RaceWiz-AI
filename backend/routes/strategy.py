from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from dependencies import get_db
from schemas.strategy import StrategySimulateRequest, StrategySimulateResponse
from services.strategy_service import StrategyService, StrategySimulationInput


router = APIRouter(prefix="/strategy", tags=["strategy"])


@router.post("/simulate", response_model=StrategySimulateResponse)
def simulate_strategy(
    payload: StrategySimulateRequest,
    db: Session = Depends(get_db),
):
    service = StrategyService(db)
    params = StrategySimulationInput(
        driver_id=int(payload.driver),
        new_compound=payload.compound,
        pit_lap=payload.pit_lap,
    )
    result = service.simulate(params)

    return {
        "session_id": result.get("session_id"),
        "driver": result["driver_id"],
        "compound": result["new_compound"],
        "pit_lap": result["pit_lap"],
        "predicted_time_delta_ms": result["predicted_time_delta_ms"],
        "risk_score": result["risk_score"],
        "strategy_viability_score": result["strategy_viability_score"],
        "details": result["details"],
    }

