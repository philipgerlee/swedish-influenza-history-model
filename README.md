# Swedish influenza historical-season model

This repository contains `gerlee-history`, an adaptive historical-season
regression model for the Swedish Influenza Forecast Hub.

## Model

For each location, the expected number of cases in an upcoming ISO week is a
non-negative linear combination of the counts in the corresponding week of
complete previous influenza seasons. The coefficients start at equal weights
and are updated from week 30 onward by non-negative ridge regression around the
equal-weight prior. The 2020/21 season is excluded. A common regularization
strength of `lambda = 1` is used for Sweden, Skåne, and Västra Götaland.

The model produces point forecasts for horizons 0--3 and uses only target and
retrospective data published by the hub. See
[the English technical description](docs/model-description-en.tex) or
[the Swedish version](docs/model-description.tex) for the mathematical details.

## Installation and tests

```bash
python -m pip install -e .
python -m pip install pytest
pytest -q
```

## Create a forecast

Run the model from the root of a local clone of
`philipgerlee/swedish_forecasting_hub`:

```bash
PYTHONPATH=/path/to/swedish-influenza-history-model/src \
python /path/to/swedish-influenza-history-model/model_template.py \
  2026-10-25 gerlee-history
```

Create all 33 retrospective 2025/26 forecast files:

```bash
PYTHONPATH=/path/to/swedish-influenza-history-model/src \
python /path/to/swedish-influenza-history-model/model_template.py \
  --all-historical gerlee-history \
  --output-root /path/to/output
```

The hub repository owns the weekly submission workflow. Whenever new target
data are merged, that workflow checks out this repository, generates the next
forecast, validates it, and opens a pull request.

## License

Code is released under the MIT License. Forecast output in the hub is released
under CC BY 4.0.
