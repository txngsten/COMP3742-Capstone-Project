"""
Student Names: Oliver Wuttke, Hans Pujalte, Shivansh Pant, Matilda Alford
Student FANs: WUTT0019, PUJA0009, PANT0108, ALFO0043
File: DataLoader.py
Date: 22-09-2026
Description:
"""

# Imports
import pandas as pd
import numpy as np

from datetime import datetime

from pandas.api.types import is_numeric_dtype

from helper_functions import (
    fetch_in_chunks_market,
    fetch_in_chunks_network,
    trim_to_curtailment_coverage,
    drop_redundant_columns,
    fill_short_gaps,
    flag_negative_curtailment,
    add_time_features,
    add_lag_features,
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

class DataLoader:
    REQUIRED = {
        'timestamp', 'energy', 'demand', 'generation_renewable',
        'generation_renewable_energy', 'renewable_proportion', 'curtailment',
    }
    MAX_GAP_HOURS = 3
    LAG_COLUMNS = ['price', 'demand', 'generation_renewable', 'curtailment']
    LAG_HOURS = [1, 24, 168]
    ROLLING_HOURS = 24

    def __init__(self, start_date: datetime, end_date: datetime, api_key: str) -> None:
        """
        Constructor for DataLoader class.
        Author: Oliver Wuttke - WUTT0019

        Args:
            start_date: Start date for fetch range
            end_date: End date for fetch range
            api_key: OpenElectricity API key
        """
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.data_set = pd.DataFrame()
        self.cleaned_data = pd.DataFrame()
        self.cleaning_report = None
        self.transformed_data = pd.DataFrame()
        self.transform_report = None
        self.modelled_data = pd.DataFrame()
        self.modelling_report = None

    def fetch(self) -> None:
        """
        Fetches both market and network data and combines into a single pandas dataframe.
        Author: Oliver Wuttke - WUTT0019
        """
        # Clear derived results when starting a new fetch.
        self.cleaned_data = pd.DataFrame()
        self.cleaning_report = None

        # Fetches market data
        market_df = fetch_in_chunks_market(
            self.start_date,
            self.end_date,
            "1h",
            30,
            self.api_key
        )

        # Fetches network data
        network_df = fetch_in_chunks_network(
            self.start_date,
            self.end_date,
            "1h",
            30,
            self.api_key
        )

        # Combines both market and network data into a dataframe
        # and assigns it to the underlying dataframe object of the class
        self.data_set = market_df.merge(network_df, on="timestamp", how="outer")

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validates measurement types, timestamps and hourly continuity.
        Author: Hans Pujalte - PUJA0009

        Args:
            data: Electricity dataframe to validate

        Raises:
            ValueError: If required columns, measurements or timestamps are invalid
        """
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
        """
        Cleans a copy of the raw dataset and records the cleaning decisions.
        Author: Hans Pujalte - PUJA0009

        Returns:
            Cleaned dataframe, also stored in self.cleaned_data

        Raises:
            ValueError: If the raw or cleaned dataset fails validation
        """

        self.cleaning_report = None
        self.cleaned_data = pd.DataFrame()
        data = self.data_set.copy(deep=True)
        self._validate_data(data)
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
        self._validate_data(data)
        self.cleaning_report = {
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
        self.cleaned_data = data
        return data

    def transform(self) -> pd.DataFrame:
        """
        Transforms the cleaned dataset into features for the machine learning sub-system.
        Author: Shivansh Pant - PANT0108

        Returns:
            Transformed dataframe, also stored in self.transformed_data

        Raises:
            ValueError: If clean() has not been run first
        """
        self.transform_report = None
        self.transformed_data = pd.DataFrame()
        if self.cleaned_data.empty:
            raise ValueError('No cleaned data found; run clean() before transform().')

        # Work on a copy so the cleaned data stays untouched.
        data = self.cleaned_data.copy(deep=True)
        rows_before = len(data)
        steps = []

        # Trim to the period where curtailment is actually recorded.
        data, coverage_start = trim_to_curtailment_coverage(data)
        steps.append({
            'step': 'trim_time_window',
            'start': coverage_start.isoformat(),
            'rows_removed': rows_before - len(data),
            'reason': 'Before source coverage, curtailment is recorded as 0, not missing, '
                      'so earlier rows would teach the model fake zeros.',
        })

        # Drop columns that duplicate others or are mostly missing.
        data, dropped = drop_redundant_columns(data)
        steps.append({
            'step': 'drop_redundant_columns',
            'columns_dropped': dropped,
            'columns_kept': [c for c in data.columns if c != 'timestamp'],
            'reason': 'Duplicate columns would count the same information twice '
                      'in distance-based models like clustering.',
        })

        # Fill short gaps so models get complete rows.
        data, filled_counts, unfilled_counts = fill_short_gaps(data, self.MAX_GAP_HOURS)
        steps.append({
            'step': 'fill_short_gaps',
            'max_gap_hours': self.MAX_GAP_HOURS,
            'values_filled': filled_counts,
            'values_left_missing': unfilled_counts,
            'rows_flagged': int(data.was_imputed.sum()),
            'reason': 'The remaining gaps are single missing hours at 02:00 when daylight saving starts, '
                      'so a straight line between neighboring hours is a fair estimate.',
        })

        # Flag hours with negative curtailment instead of removing them.
        data, negative_counts = flag_negative_curtailment(data)
        steps.append({
            'step': 'flag_negative_curtailment',
            'negative_values': negative_counts,
            'rows_flagged': int(data.curtailment_negative.sum()),
            'renewable_proportion_above_100': int(data.renewable_proportion.gt(100).sum()),
            'reason': "Curtailment can't really be negative, but keep source values "
                      'and flag these hours as suspect.',
        })

        # Add time of day, day of week, and month features.
        data, time_columns = add_time_features(data)
        steps.append({
            'step': 'add_time_features',
            'columns_added': time_columns,
            'reason': "Demand and prices follow daily, weekly, and yearly cycles; "
                      "sine and cosine keep each cycle's ends close together.",
        })

        # Add lag and rolling features from past hours.
        data, lag_columns, warmup_rows = add_lag_features(
            data,
            self.LAG_COLUMNS,
            self.LAG_HOURS,
            self.ROLLING_HOURS
        )
        steps.append({
            'step': 'add_lag_features',
            'columns_added': lag_columns,
            'rows_dropped_for_history': warmup_rows,
            'reason': 'Electricity data depends on recent history; '
                      'use only past values so the model never sees the future.',
        })

        # Store a report of what the transform did.
        self.transform_report = {
            'rows_before': rows_before,
            'rows_after': len(data),
            'first_timestamp': data.timestamp.min().isoformat(),
            'last_timestamp': data.timestamp.max().isoformat(),
            'steps': steps,
        }
        self.transformed_data = data
        return data

    def exploratory_modelling(self) -> pd.DataFrame:
        """
        Performs exploratory modelling on the transformed dataset.

        Author: Matilda Alford - ALFO0043
        
                Returns:
                    transformed dataframe with anomaly and exploratory cluster labels added
        
                Raises:
                    ValueError: If transform() has not been run first
        """

        if self.transformed_data.empty:
            raise ValueError('No transformed data found; run transform() before exploratory_modelling().')

        data = self.transformed_data.copy(deep=True)

        selected_features = [
            "curtailment",
            "curtailment_wind",
            "demand",
            "flow_exports",
            "flow_imports",
            "generation_renewable",
            "price",
            "renewable_proportion",
            "emissions",
        ]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data[selected_features])

        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_scaled)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

        data['cluster'] = kmeans.fit_predict(X_pca)

        isolation_forest = IsolationForest(contamination=0.02, random_state=42)

        data['anomaly'] = isolation_forest.fit_predict(X_scaled)

        self.modelling_report = {
            "features_selected": selected_features,
            "pca": 5,
            "kmeans_clusters": 3,
            "contamination": 0.02,
            "cluster_counts": {int(k): int(v) for k, v in data["cluster"].value_counts().sort_index().items()},
            "anomaly_counts": {int(k): int(v) for k, v in data["anomaly"].value_counts().sort_index().items()},
        }

        self.modelled_data = data
        return data

