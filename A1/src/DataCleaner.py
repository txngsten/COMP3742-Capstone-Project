"""Validated cleaning of the NSW hourly electricity dataset.

Missing measurements and unusual source values are retained. The 2000
energy correction is inferred, not confirmed for this historical range upstream.
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


class DataCleaner:
    """Clean raw input on a private copy and record decisions in report."""

    REQUIRED = {
        'timestamp', 'energy', 'demand', 'generation_renewable',
        'generation_renewable_energy', 'renewable_proportion', 'curtailment',
    }

    def __init__(self, data: pd.DataFrame):
        self._raw = data.copy(deep=True)
        self.report = None

    def _validate(self, data):
        missing = self.REQUIRED.difference(data.columns)
        if missing:
            raise ValueError(f'Missing required columns: {sorted(missing)}')
        
        if data.empty:
            raise ValueError('The dataset is empty.')
        
        timestamps = data['timestamp']
        if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
            raise ValueError('timestamp must contain timezone-aware datetimes.')
        if timestamps.isna().any() or timestamps.duplicated().any():
            raise ValueError('Missing or duplicate timestamps require investigation.')
        
        for column in data.columns.drop('timestamp'):
            if not is_numeric_dtype(data[column]):
                raise ValueError(f'{column} must be numeric.')
            if data[column].isin([np.inf, -np.inf]).any():
                raise ValueError(f'{column} contains infinite values.')
            
        ordered = timestamps.sort_values()
        if not ordered.diff().dropna().eq(pd.Timedelta(hours=1)).all():
            raise ValueError('Hourly timeline has gaps or off-grid timestamps; recover or investigate them first.')

    def clean(self) -> pd.DataFrame:
        """Return cleaned data; repeated calls always start from original input."""

        self.report = None
        data = self._raw.copy(deep=True)
        self._validate(data)
        corrections = []

        # Apply the documented notebook correction to raw 2000 observations.
        local = data['timestamp'].dt.tz_convert('Etc/GMT-10')
        affected = local.dt.year.eq(2000)
        column = 'generation_renewable_energy'
        before = data[column].copy()
        data.loc[affected, column] = before.loc[affected] * 1000
        changed = affected & before.notna() & before.ne(data[column])
        if affected.any():
            corrections.append({
                'column': column, 'year_in_fixed_aest': 2000, 'factor': 1000,
                'changed_values': int(changed.sum()),
                'status': 'Inferred correction; exact upstream historical cause unconfirmed',
                'reference': 'https://github.com/opennem/opennem/pull/606',
            })

        # Validate data again
        data = data.sort_values('timestamp').reset_index(drop=True)
        self._validate(data)
        self.report = {
            'rows': len(data),
            'first_timestamp': data.timestamp.min().isoformat(),
            'last_timestamp': data.timestamp.max().isoformat(),
            'corrections': corrections,
            'missing_counts': {c: int(v) for c, v in data.isna().sum().items()},
            'negative_curtailment_retained': int(data.curtailment.lt(0).sum()),
            'renewable_proportions_above_100_retained': int(data.renewable_proportion.gt(100).sum()),
            'limitations': [
                'Missing measurements retained; no imputation performed.',
                'Negative curtailment and suspicious renewable proportions retained.',
                'Historical curtailment zeros are not verified evidence of no curtailment.',
                'Battery and import coverage varies over time.',
                'Related energy columns remain unchanged pending separate verification.',
            ],
        }
        return data
