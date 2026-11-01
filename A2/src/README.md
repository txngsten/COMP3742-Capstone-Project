# Hybrid network-stress system

This directory contains the reproducible training and inference entry points.

## Requirements

Install the packages in `requirements.txt` and install SWI-Prolog so `pyswip`
can locate its runtime. The transformed parquet is not packaged with the
submission; generate it using A1 and pass its path with `--data` if necessary.

## Train and save the final model

```powershell
.\.venv\Scripts\python.exe A2\src\train_model.py
```

The pipeline fits target thresholds on the chronological training period,
compares Random Forest and XGBoost on validation data, refits the selected
algorithm on training plus validation data, and evaluates it once on the
untouched test period. Artifacts are saved under `A2/models`.

## Run hybrid inference

```powershell
.\.venv\Scripts\python.exe A2\src\inference.py `
  --timestamp "2026-09-23 23:00:00+10:00"
```

An optional externally sourced reserve margin can be supplied in basis points
using `--reserve-margin-bps 450` for 4.5%. It is not derived from generation
output. The integer basis-point interface also avoids a Windows PySwip issue
with decimal command-line arguments.

## Run symbolic-layer tests

```powershell
.\.venv\Scripts\python.exe -m unittest A2.src.test_hybrid -v
```

## Known limitations

- The target is a regional stress proxy, not a direct engineering measurement.
- Historical timestamp selection is a development demonstration. Live
  inference requires the A1 transformations and at least 168 hours of history.
- Prolog thresholds are development values requiring domain validation.
- Recommendations are advisory and require authorised human approval.
