"""Historical-season regression using the Swedish hub's Python starter program.

Run from the repository root:

    python submission-tools/model_template.py YYYY-MM-DD team-model
    python submission-tools/model_template.py --all-historical team-model

The program selects the correct retrospective input when one exists. For a
live round it reads target-data/time-series.csv. In both cases, observations
after the Sunday preceding reference_date are removed before the model runs.
The surrounding file handling is copied from the hub template. The replaced
``forecast_model`` body delegates the statistical calculation to the
separately tested ``influenza_history`` package.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from influenza_history import forecast_rows

LOCATIONS = ("SE", "SE-M", "SE-O")
HORIZONS = (0, 1, 2, 3)
TARGET = "weekly incident influenza cases"
OUTPUT_COLUMNS = (
    "reference_date",
    "target",
    "horizon",
    "location",
    "output_type",
    "output_type_id",
    "value",
)
MODEL_PATTERN = re.compile(r"^[A-Za-z0-9_+]+-[A-Za-z0-9_+]+$")
HISTORICAL_START = date.fromisocalendar(2025, 40, 7)
HISTORICAL_END = date.fromisocalendar(2026, 20, 7)
HISTORICAL_MANIFEST = Path("retrospective-data/2025-2026/manifest.csv")


# ---------------------------------------------------------------------------
# MODEL-SPECIFIC FUNCTION BODY
# ---------------------------------------------------------------------------
def forecast_model(
    data: list[dict[str, str]], reference_date: date
) -> list[dict[str, Any]]:
    """Return expected weekly cases for all locations and horizons 0--3."""
    return forecast_rows(data, reference_date, hub_root=Path.cwd())


# ---------------------------------------------------------------------------
# HUB FILE HANDLING — NORMALLY NO CHANGES ARE NEEDED BELOW THIS LINE
# ---------------------------------------------------------------------------
def default_input_path(reference_date: date) -> Path:
    retrospective = (
        Path("retrospective-data")
        / "2025-2026"
        / "rounds"
        / f"{reference_date.isoformat()}.csv"
    )
    if retrospective.is_file():
        return retrospective
    return Path("target-data/time-series.csv")


def read_input(path: Path, reference_date: date) -> list[dict[str, str]]:
    cutoff = reference_date - timedelta(days=7)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"location", "target_end_date", "value", "status"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Input data is missing columns: {missing}")
        rows = []
        for row in reader:
            try:
                target_end_date = date.fromisoformat(row["target_end_date"])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid target_end_date {row['target_end_date']!r}"
                ) from exc
            if target_end_date <= cutoff:
                rows.append(row)
    if not rows:
        raise ValueError(f"No input observations are available through {cutoff}")
    return rows


def validate_forecasts(forecasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not forecasts:
        raise ValueError("The model returned no forecasts")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    locations: set[str] = set()
    for number, forecast in enumerate(forecasts, start=1):
        try:
            location = str(forecast["location"])
            horizon = int(forecast["horizon"])
            value = float(forecast["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Forecast {number} must contain location, horizon and numeric value"
            ) from exc
        if location not in LOCATIONS:
            raise ValueError(f"Unsupported location {location!r}")
        if horizon not in HORIZONS:
            raise ValueError(f"Unsupported horizon {horizon!r}")
        if not math.isfinite(value) or value < 0:
            raise ValueError("Forecast values must be finite and at least zero")
        key = (location, horizon)
        if key in seen:
            raise ValueError(f"Duplicate forecast for {location}, horizon {horizon}")
        seen.add(key)
        locations.add(location)
        normalized.append(
            {"location": location, "horizon": horizon, "value": value}
        )
    expected = {(location, horizon) for location in locations for horizon in HORIZONS}
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"Every submitted location needs horizons 0-3; missing {missing}")
    return sorted(
        normalized,
        key=lambda row: (LOCATIONS.index(row["location"]), row["horizon"]),
    )


def write_submission(
    path: Path,
    reference_date: date,
    forecasts: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for forecast in forecasts:
            writer.writerow(
                {
                    "reference_date": reference_date.isoformat(),
                    "target": TARGET,
                    "horizon": forecast["horizon"],
                    "location": forecast["location"],
                    "output_type": "mean",
                    "output_type_id": "",
                    "value": forecast["value"],
                }
            )


def historical_reference_dates(path: Path = HISTORICAL_MANIFEST) -> list[date]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "reference_date" not in (reader.fieldnames or ()):
            raise ValueError("Historical manifest has no reference_date column")
        result = [date.fromisoformat(row["reference_date"]) for row in reader]
    if len(result) != 33 or len(set(result)) != 33:
        raise ValueError("Historical manifest must contain 33 unique rounds")
    if result != sorted(result) or any(value.weekday() != 6 for value in result):
        raise ValueError("Historical manifest dates must be ordered Sundays")
    return result


def prepare_round(
    reference_date: date,
    *,
    input_path: Path,
) -> list[dict[str, Any]]:
    data = read_input(input_path, reference_date)
    forecasts = validate_forecasts(forecast_model(data, reference_date))
    submitted_locations = {forecast["location"] for forecast in forecasts}
    if (
        HISTORICAL_START <= reference_date <= HISTORICAL_END
        and submitted_locations != set(LOCATIONS)
    ):
        raise ValueError("Historical rounds require forecasts for all locations")
    return forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "arguments",
        nargs="+",
        metavar="ARG",
        help="YYYY-MM-DD and model_id, or only model_id with --all-historical",
    )
    parser.add_argument(
        "--all-historical",
        action="store_true",
        help="Run every round listed in the 2025/2026 manifest",
    )
    parser.add_argument("--input-data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("model-output"))
    args = parser.parse_args()

    if args.all_historical:
        if len(args.arguments) != 1:
            parser.error("--all-historical requires exactly one model_id")
        if args.input_data or args.output:
            parser.error("--input-data and --output cannot be used with --all-historical")
        model_id = args.arguments[0]
        try:
            reference_dates = historical_reference_dates()
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Could not read historical manifest: {exc}") from exc
    else:
        if len(args.arguments) != 2:
            parser.error("provide reference_date and model_id")
        reference_date_text, model_id = args.arguments
        try:
            reference_date = date.fromisoformat(reference_date_text)
        except ValueError:
            parser.error("reference_date must use YYYY-MM-DD format")
        if reference_date.weekday() != 6:
            parser.error("reference_date must be a Sunday")
        reference_dates = [reference_date]

    if not MODEL_PATTERN.fullmatch(model_id):
        parser.error("model_id must have the form team-model")

    try:
        prepared = []
        for reference_date in reference_dates:
            input_path = args.input_data or default_input_path(reference_date)
            output_path = args.output or (
                args.output_root
                / model_id
                / f"{reference_date.isoformat()}-{model_id}.csv"
            )
            prepared.append(
                (
                    reference_date,
                    input_path,
                    output_path,
                    prepare_round(reference_date, input_path=input_path),
                )
            )
        for reference_date, input_path, output_path, forecasts in prepared:
            write_submission(output_path, reference_date, forecasts)
            print(f"Read {input_path}")
            print(f"Wrote {output_path}")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not create forecast: {exc}") from exc
    if args.all_historical:
        print(f"Wrote {len(prepared)} historical forecast files")


if __name__ == "__main__":
    main()
