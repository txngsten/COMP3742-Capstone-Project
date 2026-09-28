# Electricity data cleaning decisions

## Scope and provenance

This document records the decisions behind [the cleaning notebook](../notebooks/data_cleaning.ipynb) and [DataCleaner](../src/DataCleaner.py). Cleaning validates and corrects the ingested data; it does not create predictive features or train models.

The reviewed snapshot contains 234,312 hourly NSW observations from 2000-01-01 00:00 through 2026-09-23 23:00, in fixed UTC+10 time. Counts below describe that snapshot and may change after future downloads.

Raw data is preserved in `data/raw/electricity_raw.parquet`. The cleaned output is `data/processed/electricity_cleaned.parquet`. The pipeline writes a per-run summary to `reports/cleaning_report.json`; charts are under `reports/figures/`. Paths are relative to A1.

## Decisions

| Finding | Decision | Status |
| --- | --- | --- |
| The old downloader skipped a day between requests, omitting 7,320 hours. | Use adjoining request windows and explicitly resolve repeated boundaries. Refetch the data. | Resolved: 234,312 expected hourly timestamps are present. |
| Duplicate or invalid timestamps and gaps could break hourly alignment. | Require timezone-aware, non-null, unique timestamps and hourly continuity. Sort chronologically. Conflicting data raises an error rather than being silently deleted. | Implemented in DataCleaner. |
| Measurement types or infinity values may be invalid. | Require numeric measurements and reject infinities. Missing measurements remain allowed. | Implemented. |
| Nine energy values and ten demand values are missing. | Retain rows and NaN values, preserving other measurements and the hourly timeline. | Retained; source recovery unresolved. |
| The nine energy gaps occur around daylight-saving transitions at 02:00 in 2000–2008. | Document a possible historical time-handling issue. Do not shift timestamps or interpolate on this evidence alone. | Cause unconfirmed. |
| Imports begin in July 2009; demand-energy coverage begins in April 2005. | Preserve unavailable history as missing, without inventing values. | Coverage limitation retained. |
| Battery data is 96.06% missing across the full snapshot and becomes sparse again in 2026. | Retain missing values. Do not assume missing means zero battery output. | Cause of later gaps unresolved. |
| Renewable energy/power monthly median ratios are approximately 0.001 in 2000 and 1 thereafter. | Multiply `generation_renewable_energy` by 1,000 for fixed-AEST year 2000, using an untouched input copy. | Inferred correction applied to 8,603 nonzero, nonmissing values in the reviewed snapshot. |
| Related renewable-with-storage energy may also require investigation. | Leave `generation_renewable_with_storage_energy` unchanged pending separate verification. | Unresolved; do not assume all energy columns are corrected. |
| Renewable proportion exceeds 100% in 1,101 observations. | Retain source values pending verification of definitions and calculation; do not clip to 100. | Unresolved and present in cleaned output. |
| Curtailment has 1,157 negative observations. | Retain source values. A negative sign alone does not establish measurement corruption. | No clipping or feature engineering. |
| Early curtailment values are zero despite limited historical source coverage. | Retain source values and document that pre-September-2020 zeros are not verified evidence of no curtailment. | Coverage limitation retained. |

## Evidence for the inferred scaling correction

Open Electricity documented an erroneous division by 1,000 affecting renewable energy and other market energy metrics in [issue #605](https://github.com/opennem/opennem/issues/605) and [fix #606](https://github.com/opennem/opennem/pull/606). The [changelog](https://docs.openelectricity.org.au/changelog/) reports a historical rebuild, but does not establish why this particular 2000 discrepancy persists.

Fresh hourly NSW samples for January 1 in 2000 and 2001 exactly matched the saved observations. The API labelled energy as MWh in both years. See the [comparison](../data/validation/energy_scale_20260928T104958038336Z/comparison.csv), [summary](../data/validation/energy_scale_20260928T104958038336Z/summary.csv), and [unit metadata](../data/validation/energy_scale_20260928T104958038336Z/metadata.json).

The monthly pattern and documented bug support the correction, but the exact historical cause and range are not independently confirmed. The correction currently runs automatically on raw 2000 observations, without a ratio guard or CLI flag. Input must therefore be raw data; reusing already-cleaned data as new input would rescale it again. Repeated `clean()` calls on the same cleaner use its original input copy and do not compound the correction.

## Validation and interpretation

- Preserve raw data and export cleaned data separately.
- Compare energy/power ratios before and after correction; the reviewed 2000 median becomes approximately 1.00042.
- Check row count, timestamp uniqueness, order, hourly continuity, and missing counts.
- Monthly missingness is measured only over the covered date range; September 2026 is a partial month.
- Histograms and boxplots in the notebook describe raw observations, not corrected distributions.
- A cleaned dataset may still contain missing values and unresolved anomalies. These limitations must accompany downstream use.

Predictive lags, rolling statistics, curtailment flags, model-based imputation, and train/test preparation are outside this cleaning stage.
