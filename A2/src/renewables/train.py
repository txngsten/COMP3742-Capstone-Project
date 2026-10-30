"""
Student Name: Matilda Alford
Student FAN: ALFO0043
File: train.py
Date: 30-10-2026
Description: Compares renewable forecasting models using validation data only, uses A2 features.py.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from .features import FORECAST_HOURS, PROJECT_DIR, TARGET

OUTPUT_DIR = PROJECT_DIR / 'A2/outputs/renewables'


def same_hour_yesterday(data: pd.DataFrame, origins: pd.Series) -> np.ndarray:
    """Predict t+h using the known observation at t+h-24.

    Author: Matilda Alford - ALFO0043
    """
    history = data.set_index('timestamp')[TARGET]
    return np.column_stack([
        history.reindex(pd.DatetimeIndex(origins) + pd.Timedelta(hours=h - 24)).to_numpy()
        for h in range(1, FORECAST_HOURS + 1)
    ])


def forecast_errors(actual: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """Calculate MAE and RMSE in MW separately for each forecast hour.

    Author: Matilda Alford - ALFO0043
    """
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError('Actual and predicted values must have matching 2D shapes.')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Evaluation requires finite actual and predicted values.')
    error = predicted - actual
    return pd.DataFrame({
        'horizon_hours': range(1, actual.shape[1] + 1),
        'mae_mw': np.abs(error).mean(axis=0),
        'rmse_mw': np.sqrt((error ** 2).mean(axis=0)),
    })


def build_models() -> dict:
    """Build reproducible tree models with bounded size for local training.

    Author: Matilda Alford - ALFO0043

    Random Forest predicts all 24 outputs together. Gradient boosting fits
    one independent regressor per horizon. Trees do not require scaling.
    """
    return {
        'random_forest': RandomForestRegressor(
            n_estimators=100, max_depth=16, min_samples_leaf=5,
            random_state=42, n_jobs=2,
        ),
        'gradient_boosting': MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=100, max_leaf_nodes=15, learning_rate=0.1,
                early_stopping=False, random_state=42,
            ),
            n_jobs=1,
        ),
    }


def compare_models(
        data: pd.DataFrame,
        splits: dict[str, pd.DataFrame],
        feature_columns: list[str],
        target_columns: list[str],
        output_dir: Path = OUTPUT_DIR,
        recent_start: str = '2019-01-01',
        models: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compare two algorithms and history windows; save the best ML model.

    Author: Matilda Alford - ALFO0043

    Select by overall validation MAE. Save reports, model and input schema.
    The held-out test split is deliberately never accessed. The saved model
    remains fitted on training data only, making validation results traceable.
    The baseline is reported alongside ML models even if it outperforms them.
    """
    from sklearn.base import clone

    validation = splits['validation'].copy()
    baseline = same_hour_yesterday(data, validation['timestamp'])
    complete = np.isfinite(baseline).all(axis=1)
    validation = validation.loc[complete]
    baseline = baseline[complete]
    if validation.empty:
        raise ValueError('No validation rows have complete baseline history.')

    actual = validation[target_columns].to_numpy()
    summaries = []
    horizon_reports = []

    def record(name, window, prediction, training_rows):
        report = forecast_errors(actual, prediction)
        report['model'] = name
        report['window'] = window
        horizon_reports.append(report)
        summaries.append({
            'model': name, 'window': window, 'training_rows': training_rows,
            'validation_rows': len(validation),
            'mae_mw': float(np.abs(prediction - actual).mean()),
            'rmse_mw': float(np.sqrt(((prediction - actual) ** 2).mean())),
        })

    record('same_hour_yesterday', 'previous_day', baseline, 0)
    train = splits['train']
    start = pd.Timestamp(recent_start, tz=train['timestamp'].dt.tz)
    windows = {'full_history': train, 'recent_history': train.loc[train.timestamp >= start]}
    best_mae = float('inf')
    best_bundle = None
    candidates = build_models() if models is None else models
    if not candidates:
        raise ValueError('At least one model is required.')

    for window_name, training in windows.items():
        if training.empty:
            raise ValueError(f'No training rows in {window_name}.')
        for model_name, estimator in candidates.items():
            print(f'Training {model_name}: {window_name}, {len(training):,} examples', flush=True)
            model = clone(estimator)
            model.fit(training[feature_columns], training[target_columns])
            prediction = model.predict(validation[feature_columns])
            record(model_name, window_name, prediction, len(training))
            if summaries[-1]['mae_mw'] < best_mae:
                best_mae = summaries[-1]['mae_mw']
                best_bundle = {
                    'model': model, 'model_name': model_name, 'window': window_name,
                    'feature_columns': feature_columns, 'target_columns': target_columns,
                    'forecast_hours': FORECAST_HOURS, 'unit': 'MW',
                    'training_first_issue': training.timestamp.min().isoformat(),
                    'training_last_issue': training.timestamp.max().isoformat(),
                    'validation_first_issue': validation.timestamp.min().isoformat(),
                    'validation_last_issue': validation.timestamp.max().isoformat(),
                    'validation_mae_mw': best_mae,
                    'validation_rows_excluded_for_baseline': int((~complete).sum()),
                    'forecast_assumption': 'Hour t is complete and its reading is available.',
                }

    summary = pd.DataFrame(summaries).sort_values('mae_mw').reset_index(drop=True)
    horizons = pd.concat(horizon_reports, ignore_index=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / 'validation_comparison.csv', index=False)
    horizons.to_csv(output_dir / 'validation_by_horizon.csv', index=False)
    joblib.dump(best_bundle, output_dir / 'selected_model.joblib')
    metadata = {key: value for key, value in best_bundle.items() if key != 'model'}
    (output_dir / 'model_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return summary, horizons, metadata
