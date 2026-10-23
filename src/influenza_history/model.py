"""Adaptive linear forecasts based on corresponding weeks of past seasons."""

from __future__ import annotations

import math
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import lsq_linear


HORIZONS = (0, 1, 2, 3)
LOCATIONS = ("SE", "SE-M", "SE-O")
COMMON_RIDGE_LAMBDA = 1.0
FIRST_COMPLETE_SEASON = 2016
EXCLUDED_FEATURE_SEASONS = frozenset({2020})
HISTORICAL_START = date.fromisocalendar(2025, 40, 7)
HISTORICAL_END = date.fromisocalendar(2026, 20, 7)


@dataclass(frozen=True)
class HistoricalLinearFit:
    coefficients: np.ndarray
    scale: float


def season_dates(start_year: int, start_week: int, end_week: int) -> list[date]:
    cursor = date.fromisocalendar(start_year, start_week, 7)
    end = date.fromisocalendar(start_year + 1, end_week, 7)
    result: list[date] = []
    while cursor <= end:
        result.append(cursor)
        cursor += timedelta(days=7)
    return result


def reference_dates(start_year: int) -> list[date]:
    return season_dates(start_year, 40, 20)


def aligned_date(feature_start_year: int, target_date: date) -> date | None:
    """Map a target ISO week to the same ISO week in a historical season."""
    iso_week = target_date.isocalendar().week
    if iso_week == 53:
        return None
    calendar_year = feature_start_year if iso_week >= 30 else feature_start_year + 1
    return date.fromisocalendar(calendar_year, iso_week, 7)


def aligned_value(
    observations: dict[tuple[str, date], float],
    location: str,
    feature_start_year: int,
    target_date: date,
) -> float:
    mapped = aligned_date(feature_start_year, target_date)
    if mapped is not None:
        return observations[(location, mapped)]
    week_52 = date.fromisocalendar(feature_start_year, 52, 7)
    week_1 = date.fromisocalendar(feature_start_year + 1, 1, 7)
    return 0.5 * (
        observations[(location, week_52)] + observations[(location, week_1)]
    )


def feature_matrix(
    observations: dict[tuple[str, date], float],
    location: str,
    feature_seasons: list[int],
    target_dates: list[date],
) -> np.ndarray:
    return np.asarray(
        [
            [aligned_value(observations, location, season, target) for season in feature_seasons]
            for target in target_dates
        ],
        dtype=float,
    )


def feature_scale(
    observations: dict[tuple[str, date], float],
    location: str,
    feature_seasons: list[int],
    target_start_year: int,
) -> float:
    matrix = feature_matrix(
        observations,
        location,
        feature_seasons,
        season_dates(target_start_year, 30, 23),
    )
    return max(1.0, float(np.sqrt(np.mean(np.square(matrix)))))


def fit_coefficients(
    observations: dict[tuple[str, date], float],
    location: str,
    feature_seasons: list[int],
    target_start_year: int,
    cutoff: date,
    ridge_lambda: float,
) -> HistoricalLinearFit:
    if not feature_seasons:
        raise ValueError("At least one historical feature season is required")
    if ridge_lambda <= 0 or not np.isfinite(ridge_lambda):
        raise ValueError("ridge_lambda must be finite and greater than zero")
    n = len(feature_seasons)
    prior_mean = np.full(n, 1.0 / n, dtype=float)
    update_dates = [
        target
        for target in season_dates(target_start_year, 30, 23)
        if target <= cutoff and (location, target) in observations
    ]
    scale = feature_scale(observations, location, feature_seasons, target_start_year)
    if not update_dates:
        return HistoricalLinearFit(prior_mean, scale)
    design = feature_matrix(observations, location, feature_seasons, update_dates) / scale
    response = np.asarray(
        [observations[(location, target)] for target in update_dates], dtype=float
    ) / scale
    penalty = math.sqrt(ridge_lambda)
    augmented_design = np.vstack((design, penalty * np.eye(n)))
    augmented_response = np.concatenate((response, penalty * prior_mean))
    fitted = lsq_linear(
        augmented_design,
        augmented_response,
        bounds=(0.0, np.inf),
        tol=1e-10,
        lsmr_tol=1e-10,
        max_iter=1000,
    )
    if not fitted.success:
        raise RuntimeError(f"Coefficient fit failed: {fitted.message}")
    return HistoricalLinearFit(np.asarray(fitted.x, dtype=float), scale)


def forecast_round(
    observations: dict[tuple[str, date], float],
    location: str,
    feature_seasons: list[int],
    target_start_year: int,
    reference_date: date,
    ridge_lambda: float,
) -> tuple[list[float], HistoricalLinearFit]:
    fit = fit_coefficients(
        observations,
        location,
        feature_seasons,
        target_start_year,
        reference_date - timedelta(days=7),
        ridge_lambda,
    )
    targets = [reference_date + timedelta(days=7 * horizon) for horizon in HORIZONS]
    design = feature_matrix(observations, location, feature_seasons, targets)
    forecasts = np.maximum(0.0, design @ fit.coefficients)
    return forecasts.tolist(), fit


def season_start_year(reference_date: date) -> int:
    """Return the first calendar year of the influenza season."""
    iso = reference_date.isocalendar()
    if iso.week >= 30:
        return iso.year
    if iso.week <= 23:
        return iso.year - 1
    raise ValueError("Forecast reference date is outside the influenza season")


def observations_from_rows(
    rows: Iterable[dict[str, str]],
) -> dict[tuple[str, date], float]:
    """Convert available hub rows to the model's observation dictionary."""
    observations: dict[tuple[str, date], float] = {}
    for row in rows:
        if row.get("status") != "available" or not row.get("value", "").strip():
            continue
        location = row.get("location", "")
        if location not in LOCATIONS:
            continue
        target = date.fromisoformat(row["target_end_date"])
        value = float(row["value"])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid incidence for {location}, {target}")
        key = (location, target)
        if key in observations and observations[key] != value:
            raise ValueError(f"Conflicting incidence for {location}, {target}")
        observations[key] = value
    return observations


def complete_feature_seasons(
    observations: dict[tuple[str, date], float], target_start_year: int
) -> list[int]:
    """Find complete earlier seasons shared by all three forecast locations."""
    return [
        season
        for season in range(FIRST_COMPLETE_SEASON, target_start_year)
        if season not in EXCLUDED_FEATURE_SEASONS
        and all(
            (location, target) in observations
            for location in LOCATIONS
            for target in season_dates(season, 30, 23)
        )
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def augment_prospective_history(
    rows: list[dict[str, str]], reference_date: date, hub_root: Path | None
) -> list[dict[str, str]]:
    """Combine live data with the hub's frozen historical archive."""
    if HISTORICAL_START <= reference_date <= HISTORICAL_END:
        return rows
    root = hub_root or Path.cwd()
    retrospective = root / "retrospective-data" / "2025-2026"
    sources = (
        _read_csv(retrospective / "rounds" / "2025-10-05.csv"),
        _read_csv(retrospective / "final-outcomes.csv"),
        rows,
    )
    cutoff = reference_date - timedelta(days=7)
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for source in sources:
        for row in source:
            target = date.fromisoformat(row["target_end_date"])
            if target <= cutoff:
                merged[(row["location"], row["target_end_date"])] = row
    return list(merged.values())


def forecast_rows_with_diagnostics(
    rows: list[dict[str, str]],
    reference_date: date,
    hub_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Generate four point forecasts per location with a common lambda."""
    data = augment_prospective_history(rows, reference_date, hub_root)
    observations = observations_from_rows(data)
    target_start_year = season_start_year(reference_date)
    feature_seasons = complete_feature_seasons(observations, target_start_year)
    if len(feature_seasons) < 3:
        raise ValueError("At least three complete historical seasons are required")

    forecasts: list[dict[str, Any]] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    for location in LOCATIONS:
        values, fit = forecast_round(
            observations,
            location,
            feature_seasons,
            target_start_year,
            reference_date,
            COMMON_RIDGE_LAMBDA,
        )
        diagnostics[location] = {
            "ridge_lambda": COMMON_RIDGE_LAMBDA,
            "feature_seasons": feature_seasons,
            "coefficient_sum": float(fit.coefficients.sum()),
            "coefficients": {
                f"{season}/{season + 1}": float(weight)
                for season, weight in zip(feature_seasons, fit.coefficients)
            },
        }
        forecasts.extend(
            {
                "location": location,
                "horizon": horizon,
                "value": values[horizon],
            }
            for horizon in HORIZONS
        )
    return forecasts, diagnostics


def forecast_rows(
    rows: list[dict[str, str]],
    reference_date: date,
    hub_root: Path | None = None,
) -> list[dict[str, Any]]:
    forecasts, _ = forecast_rows_with_diagnostics(rows, reference_date, hub_root)
    return forecasts
