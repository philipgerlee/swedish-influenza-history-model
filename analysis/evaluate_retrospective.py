"""Evaluate an adaptive historical-season linear model on 2025/2026.

For each location, the expected count in week t is a non-negative linear
combination of counts in the corresponding week of previous seasons.  The
coefficients start at 1/n and are updated as observations from the current
season arrive.  The update is non-negative ridge regression around the equal-
weight prior, i.e. the MAP estimate in a Gaussian linear model with a truncated
normal prior.  Only point forecasts are produced.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import fmean

from influenza_history.model import (
    forecast_round,
    reference_dates,
    season_dates,
)


MODEL_ID = "analysis-historical-linear"
LOCATIONS = ("SE", "SE-M", "SE-O")
HORIZONS = (0, 1, 2, 3)
LAMBDA_CANDIDATES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
FIRST_COMPLETE_SEASON = 2016
HELD_OUT_SEASON = 2025
MIN_CALIBRATION_FEATURE_SEASONS = 3


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_complete_season(
    observations: dict[tuple[str, date], float], start_year: int
) -> bool:
    return all(
        (location, target) in observations
        for location in LOCATIONS
        for target in season_dates(start_year, 30, 23)
    )


def feature_seasons_before(
    complete_seasons: list[int], target_start_year: int, excluded: set[int]
) -> list[int]:
    return [
        season
        for season in complete_seasons
        if season < target_start_year and season not in excluded
    ]


def calibration_scores(
    observations: dict[tuple[str, date], float],
    complete_seasons: list[int],
    excluded: set[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for location in LOCATIONS:
        for ridge_lambda in LAMBDA_CANDIDATES:
            errors: list[float] = []
            target_seasons: list[int] = []
            for target_start_year in complete_seasons:
                if target_start_year >= HELD_OUT_SEASON or target_start_year in excluded:
                    continue
                features = feature_seasons_before(
                    complete_seasons, target_start_year, excluded
                )
                if len(features) < MIN_CALIBRATION_FEATURE_SEASONS:
                    continue
                target_seasons.append(target_start_year)
                for reference_date in reference_dates(target_start_year):
                    forecasts, _ = forecast_round(
                        observations,
                        location,
                        features,
                        target_start_year,
                        reference_date,
                        ridge_lambda,
                    )
                    errors.extend(
                        abs(forecast - observations[(location, target)])
                        for forecast, target in zip(
                            forecasts,
                            [
                                reference_date + timedelta(days=7 * horizon)
                                for horizon in HORIZONS
                            ],
                        )
                    )
            rows.append(
                {
                    "location": location,
                    "ridge_lambda": ridge_lambda,
                    "mae": round(fmean(errors), 8),
                    "n": len(errors),
                    "target_seasons": ";".join(str(year) for year in sorted(set(target_seasons))),
                }
            )
    return rows


def select_lambdas(scores: list[dict[str, object]]) -> dict[str, float]:
    """Select one common lambda by mean calibration MAE across locations."""
    common_lambda = min(
        LAMBDA_CANDIDATES,
        key=lambda candidate: (
            fmean(
                float(row["mae"])
                for row in scores
                if float(row["ridge_lambda"]) == candidate
            ),
            -candidate,
        ),
    )
    return {location: float(common_lambda) for location in LOCATIONS}


def evaluate_held_out(
    observations: dict[tuple[str, date], float],
    complete_seasons: list[int],
    excluded: set[int],
    selected_lambdas: dict[str, float],
    variant: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    matched: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    features = feature_seasons_before(complete_seasons, HELD_OUT_SEASON, excluded)
    for reference_date in reference_dates(HELD_OUT_SEASON):
        for location in LOCATIONS:
            forecasts, fit = forecast_round(
                observations,
                location,
                features,
                HELD_OUT_SEASON,
                reference_date,
                selected_lambdas[location],
            )
            weight_rows.append(
                {
                    "variant": variant,
                    "reference_date": reference_date.isoformat(),
                    "location": location,
                    "ridge_lambda": selected_lambdas[location],
                    "coefficient_sum": round(float(fit.coefficients.sum()), 8),
                    **{
                        f"weight_{season}_{season + 1}": round(float(weight), 8)
                        for season, weight in zip(features, fit.coefficients)
                    },
                }
            )
            for horizon, forecast in enumerate(forecasts):
                target = reference_date + timedelta(days=7 * horizon)
                observed = observations[(location, target)]
                error = forecast - observed
                matched.append(
                    {
                        "model_id": MODEL_ID,
                        "variant": variant,
                        "reference_date": reference_date.isoformat(),
                        "target_end_date": target.isoformat(),
                        "location": location,
                        "horizon": horizon,
                        "forecast": round(forecast, 8),
                        "observed": observed,
                        "error": round(error, 8),
                        "absolute_error": round(abs(error), 8),
                        "squared_error": round(error**2, 8),
                    }
                )
    return matched, weight_rows


def grouped_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["location"]), int(row["horizon"]))].append(row)
    metrics: list[dict[str, object]] = []
    for location in LOCATIONS:
        for horizon in HORIZONS:
            group = grouped[(location, horizon)]
            metrics.append(
                {
                    "location": location,
                    "horizon": horizon,
                    "n": len(group),
                    "mae": round(fmean(float(row["absolute_error"]) for row in group), 8),
                    "bias": round(fmean(float(row["error"]) for row in group), 8),
                    "rmse": round(
                        math.sqrt(fmean(float(row["squared_error"]) for row in group)),
                        8,
                    ),
                }
            )
    return metrics


def load_observations(hub_root: Path) -> dict[tuple[str, date], float]:
    retrospective = hub_root / "retrospective-data" / "2025-2026"
    observations: dict[tuple[str, date], float] = {}
    for source in (
        retrospective / "rounds" / "2025-10-05.csv",
        retrospective / "final-outcomes.csv",
    ):
        for row in read_csv(source):
            if row["status"] != "available" or not row["value"].strip():
                continue
            observations[(row["location"], date.fromisoformat(row["target_end_date"]))] = float(
                row["value"]
            )
    return observations


def run_variant(
    output: Path,
    observations: dict[tuple[str, date], float],
    complete_seasons: list[int],
    excluded: set[int],
    variant: str,
) -> dict[str, object]:
    scores = calibration_scores(observations, complete_seasons, excluded)
    selected = select_lambdas(scores)
    matched, weights = evaluate_held_out(
        observations, complete_seasons, excluded, selected, variant
    )
    metrics = grouped_metrics(matched)
    write_csv(output / f"calibration-scores-{variant}.csv", scores)
    write_csv(output / f"matched-forecasts-{variant}.csv", matched)
    write_csv(output / f"metrics-{variant}.csv", metrics)
    write_csv(output / f"weights-{variant}.csv", weights)
    return {
        "variant": variant,
        "excluded_seasons": sorted(excluded),
        "feature_seasons": feature_seasons_before(
            complete_seasons, HELD_OUT_SEASON, excluded
        ),
        "selected_lambdas": selected,
        "lambda_selection": "common across locations by mean calibration MAE",
        "matched_forecasts": len(matched),
        "overall_mae": round(
            fmean(float(row["absolute_error"]) for row in matched), 8
        ),
        "overall_bias": round(fmean(float(row["error"]) for row in matched), 8),
    }


def main() -> None:
    model_root = Path(__file__).resolve().parents[1]
    hub_root = model_root.parent / "swedish_forecasting_hub"
    observations = load_observations(hub_root)
    complete_seasons = [
        season
        for season in range(FIRST_COMPLETE_SEASON, HELD_OUT_SEASON + 1)
        if is_complete_season(observations, season)
    ]
    if HELD_OUT_SEASON not in complete_seasons:
        raise RuntimeError("Held-out season is incomplete")
    output = model_root / "analysis-output" / "historical-linear"
    output.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_variant(
            output,
            observations,
            complete_seasons,
            {2020},
            "exclude-2020-2021",
        ),
        run_variant(
            output,
            observations,
            complete_seasons,
            set(),
            "include-2020-2021",
        ),
    ]
    summary = {
        "model_id": MODEL_ID,
        "held_out_season": "2025/2026",
        "complete_seasons": complete_seasons,
        "lambda_candidates": list(LAMBDA_CANDIDATES),
        "variants": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
