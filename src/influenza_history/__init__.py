"""Adaptive historical-season forecasts for the Swedish Influenza Forecast Hub."""

from .model import forecast_rows, forecast_rows_with_diagnostics

__all__ = ["forecast_rows", "forecast_rows_with_diagnostics"]
