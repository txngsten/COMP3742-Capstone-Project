"""Compare fresh NSW hourly renewable metrics with the saved raw dataset.

Run with the project Python environment. Makes two one-day API requests.
Writes audit files to A1/data/validation; never changes the raw dataset.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openelectricity import OEClient
from openelectricity.types import MarketMetric

PROJECT = Path(__file__).resolve().parents[1]
METRICS = ['generation_renewable', 'generation_renewable_energy']
AEST = timezone(timedelta(hours=10))


def compare(saved, fresh):
    """Align observations by instant, retain unmatched rows, and compute ratios."""
    saved = saved[['timestamp', *METRICS]].copy()
    fresh = fresh[['timestamp', *METRICS]].copy()
    for frame in [saved, fresh]:
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], utc=True)
    result = fresh.merge(saved, on='timestamp', how='outer',
                         suffixes=('_fresh', '_saved'),
                         validate='one_to_one', indicator=True)
    for metric in METRICS:
        denominator = result[f'{metric}_saved']
        result[f'{metric}_fresh_to_saved'] = (
            result[f'{metric}_fresh'] / denominator.where(denominator.ne(0))
        )
    for source in ['saved', 'fresh']:
        power = result[f'generation_renewable_{source}']
        result[f'energy_power_ratio_{source}'] = (
            result[f'generation_renewable_energy_{source}']
            / power.where(power.ne(0))
        )
    return result.sort_values('timestamp')


def main():
    # Explicit locations make this runnable from either the repo or src directory.
    for location in [PROJECT / 'src' / '.env', PROJECT.parent / '.env']:
        load_dotenv(location, override=False)
    key = os.getenv('OPENELECTRICITY_API_KEY')
    if not key:
        raise SystemExit('OPENELECTRICITY_API_KEY is missing from the environment or project .env.')

    saved = pd.read_parquet(PROJECT / 'data/raw/electricity_raw.parquet', engine='pyarrow')
    saved['timestamp'] = pd.to_datetime(saved['timestamp'], utc=True)
    client = OEClient(key)
    tables, metadata, summaries = [], [], []
    for year in [2000, 2001]:
        start = datetime(year, 1, 1)
        end = start + timedelta(days=1)
        # API query boundaries are network-local; compare in UTC internally.
        lower = pd.Timestamp(start, tz=AEST).tz_convert('UTC')
        upper = pd.Timestamp(end, tz=AEST).tz_convert('UTC')
        response = client.get_market(
            network_code='NEM', network_region='NSW1', interval='1h',
            metrics=[MarketMetric.GENERATION_RENEWABLE,
                     MarketMetric.GENERATION_RENEWABLE_ENERGY],
            date_start=start, date_end=end,
        )
        records = []
        for series in response.data:
            metric = getattr(series.metric, 'value', series.metric)
            metadata.append({'sample_year': year, 'metric': str(metric),
                             'unit': str(series.unit)})
            for result in series.results:
                for point in result.data:
                    records.append({'timestamp': point.timestamp,
                                    'metric': str(metric), 'value': point.value})
        if not records:
            raise ValueError(f'No observations returned for {year}')
        long = pd.DataFrame(records)
        long['timestamp'] = pd.to_datetime(long['timestamp'], utc=True)
        long = long.loc[long.timestamp.ge(lower) & long.timestamp.lt(upper)]
        if long.duplicated(['timestamp', 'metric']).any():
            raise ValueError('API returned duplicate timestamp/metric pairs; inspect grouping.')
        fresh = long.pivot(index='timestamp', columns='metric', values='value').reset_index()
        fresh.columns.name = None
        original = saved.loc[saved.timestamp.ge(lower) & saved.timestamp.lt(upper)]
        comparison = compare(original, fresh)
        comparison.insert(0, 'sample_year', year)
        tables.append(comparison)
        summary = {'sample_year': year, 'expected_hours': 24,
                   'fresh_hours': len(fresh), 'saved_hours': len(original),
                   'matched_hours': int(comparison['_merge'].eq('both').sum())}
        for name in [f'{m}_fresh_to_saved' for m in METRICS] + [
                'energy_power_ratio_saved', 'energy_power_ratio_fresh']:
            values = comparison[name].replace([np.inf, -np.inf], np.nan).dropna()
            summary[name + '_valid_count'] = len(values)
            for stat in ['min', 'median', 'max']:
                summary[name + '_' + stat] = float(getattr(values, stat)()) if len(values) else None
        summaries.append(summary)
        print(f'{year}: {summary["matched_hours"]}/24 hours matched; '
              f'energy fresh/saved median = {summary["generation_renewable_energy_fresh_to_saved_median"]}; '
              f'fresh energy/power median = {summary["energy_power_ratio_fresh_median"]}')

    output = PROJECT / 'data/validation' / datetime.now(timezone.utc).strftime('energy_scale_%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(tables, ignore_index=True).to_csv(output / 'comparison.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False)
    (output / 'metadata.json').write_text(json.dumps({
        'fetched_at_utc': datetime.now(timezone.utc).isoformat(),
        'region': 'NSW1', 'interval': '1h', 'source_units': metadata,
        'note': 'Two January 1 samples only; not proof of full-year correction.'
    }, indent=2) + '\n')
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()
