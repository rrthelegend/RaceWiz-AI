from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.lap import Lap
from app.models.stint import Stint


@dataclass
class StrategySimulationInput:
    session_id: int
    driver_id: int
    new_compound: str
    pit_lap: int


class StrategyService:
    """
    Heuristic strategy simulator.
    This is intentionally simple but structured so that
    ML models can be slotted in later.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def simulate(self, params: StrategySimulationInput) -> dict[str, Any]:
        """
        Simulate a single pit stop where the driver switches to
        `new_compound` on lap `pit_lap`.

        The baseline is the driver's historical pace in this session
        on all compounds; we estimate the effect of the tyre change
        by comparing compound‑specific averages.
        """
        laps_df = self._laps_for_driver(params.session_id, params.driver_id)
        if laps_df.empty:
            return {
                "session_id": params.session_id,
                "driver_id": params.driver_id,
                "new_compound": params.new_compound,
                "pit_lap": params.pit_lap,
                "predicted_time_delta_ms": 0.0,
                "risk_score": 0.0,
                "strategy_viability_score": 0.0,
                "details": {"reason": "no_lap_data"},
            }

        # Determine race length and baseline total time from historical data
        total_laps = int(laps_df["lap_number"].max())

        baseline_total_ms = float(laps_df["lap_time_ms"].sum())

        # Compound‑specific pace
        compound_stats = (
            laps_df.groupby("compound")["lap_time_ms"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_lap_ms", "count": "lap_count"})
        )

        # If we have historical data for the requested compound, use it;
        # otherwise, fall back to overall average.
        if params.new_compound in compound_stats.index:
            new_compound_pace_ms = float(compound_stats.loc[params.new_compound, "avg_lap_ms"])
        else:
            new_compound_pace_ms = float(laps_df["lap_time_ms"].mean())

        # Estimate baseline post‑pit pace using the compound before pit
        pre_pit_laps = laps_df[laps_df["lap_number"] < params.pit_lap]
        if pre_pit_laps.empty:
            baseline_post_pit_ms = float(laps_df["lap_time_ms"].mean())
        else:
            last_compound = pre_pit_laps.iloc[-1]["compound"]
            if last_compound in compound_stats.index:
                baseline_post_pit_ms = float(compound_stats.loc[last_compound, "avg_lap_ms"])
            else:
                baseline_post_pit_ms = float(laps_df["lap_time_ms"].mean())

        laps_after_pit = max(0, total_laps - params.pit_lap + 1)

        baseline_post_pit_total_ms = baseline_post_pit_ms * laps_after_pit
        simulated_post_pit_total_ms = new_compound_pace_ms * laps_after_pit

        predicted_time_delta_ms = baseline_post_pit_total_ms - simulated_post_pit_total_ms

        # Risk: compare requested stint length vs historical stint lengths
        risk_score = self._estimate_risk(
            session_id=params.session_id,
            compound=params.new_compound,
            desired_stint_length=laps_after_pit,
        )

        # Simple viability heuristic scaled 0‑100
        # Positive time_delta and lower risk -> higher score.
        time_gain_factor = max(-30_000.0, min(30_000.0, predicted_time_delta_ms)) / 30_000.0
        viability = max(
            0.0,
            min(
                100.0,
                50.0 + time_gain_factor * 40.0 - risk_score * 10.0,
            ),
        )

        return {
            "session_id": params.session_id,
            "driver_id": params.driver_id,
            "new_compound": params.new_compound,
            "pit_lap": params.pit_lap,
            "predicted_time_delta_ms": float(predicted_time_delta_ms),
            "risk_score": float(risk_score),
            "strategy_viability_score": float(viability),
            "details": {
                "total_laps": total_laps,
                "laps_after_pit": laps_after_pit,
                "baseline_post_pit_avg_lap_ms": float(baseline_post_pit_ms),
                "simulated_post_pit_avg_lap_ms": float(new_compound_pace_ms),
            },
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _laps_for_driver(self, session_id: int, driver_id: int) -> pd.DataFrame:
        laps = list(
            self.db.scalars(
                select(Lap).where(
                    Lap.session_id == session_id,
                    Lap.driver_id == driver_id,
                )
            ).all()
        )
        if not laps:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "lap_id": l.id,
                    "lap_number": l.lap_number,
                    "lap_time_ms": l.lap_time_ms,
                    "compound": l.compound,
                }
                for l in laps
            ]
        )

    def _estimate_risk(
        self, session_id: int, compound: str, desired_stint_length: int
    ) -> float:
        stints = list(
            self.db.scalars(
                select(Stint).where(
                    Stint.session_id == session_id,
                    Stint.compound == compound,
                )
            ).all()
        )
        if not stints:
            # No reference data -> higher uncertainty / risk
            return 0.7

        lengths = [s.stint_length for s in stints if s.stint_length is not None]
        if not lengths:
            return 0.6

        avg_length = sum(lengths) / len(lengths)

        # Risk grows as we exceed typical stint length
        if desired_stint_length <= avg_length:
            return 0.2

        # Simple linear scaling beyond average, capped at 1.0
        overrun_ratio = (desired_stint_length - avg_length) / max(avg_length, 1.0)
        return float(max(0.2, min(1.0, 0.2 + overrun_ratio)))

