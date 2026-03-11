from collections.abc import Sequence

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.models.driver import Driver as DriverModel
from dependencies import get_db
from schemas.driver import Driver as DriverSchema


router = APIRouter(prefix="/drivers", tags=["drivers"])


@router.get("/", response_model=list[DriverSchema])
def list_drivers(
    team_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
) -> Sequence[DriverModel]:
    query = db.query(DriverModel)
    if team_id is not None:
        query = query.filter(DriverModel.team_id == team_id)
    return query.order_by(DriverModel.code).all()


@router.get("/{driver_id}", response_model=DriverSchema)
def get_driver(driver_id: int, db: Session = Depends(get_db)) -> DriverModel:
    driver = db.get(DriverModel, driver_id)
    if not driver:
        raise HTTPException(status_code=404, detail="Driver not found")
    return driver

