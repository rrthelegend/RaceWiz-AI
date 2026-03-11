from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from dependencies import get_db
from services.strategy_service import StrategyService, StrategySimulationInput


class StrategySimulationRequest(BaseModel):
    session_id: int = Field(..., description="Race session identifier")
    driver_id: int = Field(..., description="Driver identifier")
    tyre_compound: str = Field(..., description="Target tyre compound, e.g. 'SOFT'")
    pit_lap: int = Field(..., description="Lap number of the pit stop")


router = APIRouter(prefix="/strategy", tags=["strategy"])


@router.post("/simulate")
def simulate_strategy(
    payload: StrategySimulationRequest,
    db: Session = Depends(get_db),
):
    service = StrategyService(db)
    params = StrategySimulationInput(
        session_id=payload.session_id,
        driver_id=payload.driver_id,
        new_compound=payload.tyre_compound.upper(),
        pit_lap=payload.pit_lap,
    )
    return service.simulate(params)

