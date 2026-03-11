from __future__ import annotations

from collections.abc import Iterable
from statistics import mean
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.lap import Lap
from app.models.stint import Stint


class AnalyticsService:
    """
    High‑level race analytics built on top of the core
    SQLAlchemy models. Most methods return plain dicts
    or lists that can be exposed directly via FastAPI.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Tyre / stint analytics
    # ------------------------------------------------------------------
    def tyre_stint_lengths(
        self, session_id: int, driver_id: int | None = None
    ) -> list[dict[str, Any]]:
        query = select(Stint).where(Stint.session_id == session_id)
        if driver_id is not None:
            query = query.where(Stint.driver_id == driver_id)

        stints: Iterable[Stint] = self.db.scalars(query).all()

        return [
            {
                "stint_id": s.id,
                "driver_id": s.driver_id,
                "session_id": s.session_id,
                "compound": s.compound,
                "start_lap": s.start_lap,
                "end_lap": s.end_lap,
                "stint_length": s.stint_length,
            }
            for s in stints
        ]

    def tyre_compound_summary(self, session_id: int) -> list[dict[str, Any]]:
        """
        Aggregate stint lengths per compound across the field.
        Useful for quick 'how long did tyres last' insights.
        """
        stmt = (
            select(
                Stint.compound,
                func.count(Stint.id).label("stint_count"),
                func.avg(Stint.stint_length).label("avg_length"),
                func.min(Stint.stint_length).label("min_length"),
                func.max(Stint.stint_length).label("max_length"),
            )
            .where(Stint.session_id == session_id)
            .group_by(Stint.compound)
        )

        rows = self.db.execute(stmt).all()
        return [
            {
                "compound": r.compound,
                "stint_count": int(r.stint_count),
                "avg_length": float(r.avg_length) if r.avg_length is not None else None,
                "min_length": int(r.min_length) if r.min_length is not None else None,
                "max_length": int(r.max_length) if r.max_length is not None else None,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Pace analytics
    # ------------------------------------------------------------------
    def lap_pace_delta(
        self, session_id: int, driver_id: int | None = None
    ) -> dict[str, Any]:
        """
        Compute lap‑by‑lap delta vs session average for the given driver
        (or for all drivers if driver_id is omitted).
        """
        query = select(Lap).where(Lap.session_id == session_id)
        if driver_id is not None:
            query = query.where(Lap.driver_id == driver_id)

        laps: list[Lap] = list(self.db.scalars(query).all())
        if not laps:
            return {"session_id": session_id, "driver_id": driver_id, "laps": []}

        df = pd.DataFrame(
            [
                {
                    "lap_id": l.id,
                    "driver_id": l.driver_id,
                    "lap_number": l.lap_number,
                    "lap_time_ms": l.lap_time_ms,
                    "compound": l.compound,
                    "stint": l.stint,
                }
                for l in laps
            ]
        )

        session_avg = df["lap_time_ms"].mean()
        df["delta_vs_session_ms"] = df["lap_time_ms"] - session_avg

        if driver_id is not None:
            driver_group = df.groupby("driver_id")["lap_time_ms"]
            driver_avg = float(driver_group.mean().iloc[0])
        else:
            driver_avg = float(session_avg)

        df["delta_vs_driver_ms"] = df["lap_time_ms"] - driver_avg

        return {
            "session_id": session_id,
            "driver_id": driver_id,
            "session_average_lap_ms": float(session_avg),
            "driver_average_lap_ms": driver_avg,
            "laps": df.to_dict(orient="records"),
        }

    # ------------------------------------------------------------------
    # Undercut / overcut & air quality heuristics
    # ------------------------------------------------------------------
    def undercut_overcut_summary(
        self, session_id: int
    ) -> list[dict[str, Any]]:
        """
        Very lightweight heuristic:
        - For each driver, look at average pace change when starting a new stint.
        - If a driver's new‑stint pace improves more than the field median and
          the stop is earlier than the median lap for that compound, flag as undercut.
        - If the stop is later than median and pace still improves strongly,
          flag as overcut.
        """
        stints: list[Stint] = list(
            self.db.scalars(
                select(Stint).where(Stint.session_id == session_id)
            ).all()
        )
        if not stints:
            return []

        # median start lap per compound
        compound_to_starts: dict[str, list[int]] = {}
        for s in stints:
            compound_to_starts.setdefault(s.compound, []).append(s.start_lap)

        compound_to_median_start = {
            c: sorted(laps)[len(laps) // 2]
            for c, laps in compound_to_starts.items()
        }

        results: list[dict[str, Any]] = []
        for s in stints:
            stint_df = self._laps_for_range(
                session_id=session_id,
                driver_id=s.driver_id,
                start_lap=s.start_lap,
                end_lap=s.end_lap,
            )
            if stint_df.empty:
                continue

            # compare first 3 laps of the stint vs last 3 laps before pit
            before_df = self._laps_for_range(
                session_id=session_id,
                driver_id=s.driver_id,
                start_lap=s.start_lap - 3,
                end_lap=s.start_lap - 1,
            )
            if before_df.empty:
                continue

            new_pace = stint_df.head(3)["lap_time_ms"].mean()
            old_pace = before_df["lap_time_ms"].mean()
            improvement = float(old_pace - new_pace)

            median_start = compound_to_median_start.get(s.compound, s.start_lap)
            strategy_type = "neutral"
            if s.start_lap < median_start and improvement > 0:
                strategy_type = "undercut"
            elif s.start_lap > median_start and improvement > 0:
                strategy_type = "overcut"

            results.append(
                {
                    "driver_id": s.driver_id,
                    "stint_id": s.id,
                    "compound": s.compound,
                    "start_lap": s.start_lap,
                    "median_start_lap_for_compound": median_start,
                    "pace_improvement_ms": improvement,
                    "strategy_type": strategy_type,
                }
            )

        return results

    def air_quality_indicators(
        self, session_id: int, driver_id: int
    ) -> dict[str, Any]:
        """
        Clean vs dirty air proxy:
        - For each lap, compare lap time to a rolling mean of that driver's
          pace. Laps significantly slower than rolling mean are flagged as
          'dirty air' candidates.
        """
        df = self._laps_for_range(
            session_id=session_id, driver_id=driver_id, start_lap=None, end_lap=None
        )
        if df.empty:
            return {"session_id": session_id, "driver_id": driver_id, "laps": []}

        df = df.sort_values("lap_number")
        df["rolling_mean_ms"] = (
            df["lap_time_ms"].rolling(window=5, min_periods=3).mean()
        )
        df["delta_vs_rolling_ms"] = df["lap_time_ms"] - df["rolling_mean_ms"]

        threshold = max(0.0, df["delta_vs_rolling_ms"].std(skipna=True) or 0.0)
        if threshold == 0:
            threshold = mean(df["lap_time_ms"]) * 0.01  # 1% of average as fallback

        df["air_quality"] = df["delta_vs_rolling_ms"].apply(
            lambda d: "dirty" if d is not None and d > threshold else "clean"
        )

        return {
            "session_id": session_id,
            "driver_id": driver_id,
            "threshold_ms": float(threshold),
            "laps": df.to_dict(orient="records"),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _laps_for_range(
        self,
        session_id: int,
        driver_id: int,
        start_lap: int | None,
        end_lap: int | None,
    ) -> pd.DataFrame:
        query = select(Lap).where(
            Lap.session_id == session_id,
            Lap.driver_id == driver_id,
        )
        if start_lap is not None:
            query = query.where(Lap.lap_number >= start_lap)
        if end_lap is not None:
            query = query.where(Lap.lap_number <= end_lap)

        laps: list[Lap] = list(self.db.scalars(query).all())
        if not laps:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "lap_id": l.id,
                    "driver_id": l.driver_id,
                    "lap_number": l.lap_number,
                    "lap_time_ms": l.lap_time_ms,
                    "compound": l.compound,
                    "stint": l.stint,
                }
                for l in laps
            ]
        )

