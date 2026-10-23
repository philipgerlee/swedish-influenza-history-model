from datetime import date

import numpy as np
import pytest

from influenza_history.model import (
    COMMON_RIDGE_LAMBDA,
    aligned_value,
    complete_feature_seasons,
    fit_coefficients,
    forecast_rows,
    forecast_round,
    augment_prospective_history,
    season_dates,
)


LOCATION = "SE"


def constant_season(observations, start_year, value):
    for target in season_dates(start_year, 30, 23):
        observations[(LOCATION, target)] = float(value)


def test_equal_weight_prior_before_current_observations():
    observations = {}
    constant_season(observations, 2023, 10)
    constant_season(observations, 2024, 20)
    fitted = fit_coefficients(
        observations,
        LOCATION,
        [2023, 2024],
        2025,
        date.fromisocalendar(2025, 29, 7),
        1.0,
    )
    np.testing.assert_allclose(fitted.coefficients, [0.5, 0.5])


def test_coefficients_update_from_week_30_and_forecast_remains_nonnegative():
    observations = {}
    constant_season(observations, 2023, 10)
    constant_season(observations, 2024, 10)
    for target in season_dates(2025, 30, 39):
        observations[(LOCATION, target)] = 20.0
    forecasts, fitted = forecast_round(
        observations,
        LOCATION,
        [2023, 2024],
        2025,
        date.fromisocalendar(2025, 40, 7),
        0.01,
    )
    assert fitted.coefficients.sum() == pytest.approx(2.0, rel=2e-3)
    np.testing.assert_allclose(forecasts, [20.0] * 4, rtol=2e-3)
    assert np.all(fitted.coefficients >= 0)


def test_week_53_is_interpolated_from_weeks_52_and_1():
    observations = {
        (LOCATION, date.fromisocalendar(2019, 52, 7)): 80.0,
        (LOCATION, date.fromisocalendar(2020, 1, 7)): 120.0,
    }
    target = date.fromisocalendar(2026, 53, 7)
    assert aligned_value(observations, LOCATION, 2019, target) == 100.0


def test_invalid_ridge_strength_is_rejected():
    observations = {}
    constant_season(observations, 2024, 10)
    with pytest.raises(ValueError, match="ridge_lambda"):
        fit_coefficients(
            observations,
            LOCATION,
            [2024],
            2025,
            date.fromisocalendar(2025, 39, 7),
            0.0,
        )


def test_operational_forecast_uses_one_common_lambda():
    observations = {}
    rows = []
    for location in ("SE", "SE-M", "SE-O"):
        for start_year, value in ((2016, 10), (2017, 20), (2018, 30)):
            for target in season_dates(start_year, 30, 23):
                observations[(location, target)] = float(value)
        for target in season_dates(2019, 30, 39):
            observations[(location, target)] = 20.0
    for (location, target), value in observations.items():
        rows.append(
            {
                "location": location,
                "target_end_date": target.isoformat(),
                "value": str(value),
                "status": "available",
            }
        )

    forecasts = forecast_rows(rows, date.fromisocalendar(2019, 40, 7))

    assert COMMON_RIDGE_LAMBDA == 1.0
    assert complete_feature_seasons(observations, 2019) == [2016, 2017, 2018]
    assert len(forecasts) == 12
    assert {row["location"] for row in forecasts} == {"SE", "SE-M", "SE-O"}


def test_prospective_history_is_augmented_from_frozen_archive(tmp_path):
    archive = tmp_path / "retrospective-data" / "2025-2026"
    rounds = archive / "rounds"
    rounds.mkdir(parents=True)
    header = "location,target_end_date,value,status\n"
    (rounds / "2025-10-05.csv").write_text(
        header + "SE,2024-10-06,10,available\n", encoding="utf-8"
    )
    (archive / "final-outcomes.csv").write_text(
        header + "SE,2026-05-17,20,available\n", encoding="utf-8"
    )
    live = [
        {
            "location": "SE",
            "target_end_date": "2026-10-18",
            "value": "30",
            "status": "available",
        }
    ]

    combined = augment_prospective_history(
        live, date.fromisocalendar(2026, 43, 7), tmp_path
    )

    assert {row["target_end_date"] for row in combined} == {
        "2024-10-06",
        "2026-05-17",
        "2026-10-18",
    }
