# OpenElectricity metric data dictionary

This document describes the OpenElectricity metrics requested by this project's
pipeline for the NEM `NSW1` region at a `1h` interval. It is based on the
OpenElectricity v4 API metric catalogue and official OpenElectricity guides,
checked on 27 September 2026.

## Interpretation notes

- **API type:** Every metric is returned as a JSON numeric value or `null`. In
  the project's wide pandas dataframe, these columns normally have the
  `float64` dtype because pandas represents missing numeric values as `NaN`.
- **Expected range:** This is a domain-based data-quality expectation, not a
  guarantee enforced by the API. A value outside an expected range should be
  flagged and investigated before it is changed or removed.
- **Missing is not zero:** A missing value means that OpenElectricity did not
  provide a usable observation for that metric and timestamp. Zero is a valid
  measured or calculated result and must not be substituted automatically.
- **Coverage varies:** Historical coverage differs by metric and source. A
  column can be structurally missing before its source or calculation became
  available. OpenElectricity reports NEM network data from 19 September 2006;
  requesting dates from 2000 therefore does not guarantee observations for all
  fields.
- **Aggregated values:** These definitions describe the region-level values
  returned by the project's network and market queries. Facility- or unit-level
  values can have different sign conventions, particularly loads and
  bidirectional batteries.
- **Time:** API query boundaries are in network-local time. The saved dataframe
  currently uses timezone-aware timestamps with the NEM `UTC+10:00` offset.

## Network metrics

Endpoint: `GET /v4/data/network/NEM`, filtered to `network_region=NSW1`.

| Field | Source and derivation | API/Pandas type | Unit | Expected range / validation rule | Meaning of a missing value |
|---|---|---|---|---|---|
| `power` | OpenElectricity aggregation of generator power observations, sourced principally from AEMO dispatch/SCADA data. It represents instantaneous output or consumption; the API catalogue uses average aggregation for wider time buckets. | `number \| null` / `float64` | MW | For this region-level generation aggregate, normally `>= 0`. At unit level, load or bidirectional battery readings may be signed. Must be finite and should be checked against plausible installed capacity. | No usable power observation was available for the timestamp/group, or all contributing source observations were absent. It does **not** mean zero output. |
| `energy` | Derived by OpenElectricity from power using trapezoidal integration: average of the current and previous power readings multiplied by interval duration. The API sums energy when aggregating intervals. | `number \| null` / `float64` | MWh | For this region-level aggregate, normally `>= 0`. At `1h`, energy should be broadly consistent with adjacent hourly power, allowing for the integration method. | Energy could not be calculated, commonly because a required power observation was unavailable, or the source interval was absent. |
| `emissions` | Calculated by OpenElectricity as generated energy multiplied by unit-specific emissions factors, then summed. Factors are sourced from AEMO, the Clean Energy Regulator, state sources, or are OpenElectricity estimates. Operational emissions only. | `number \| null` / `float64` | tCO2e (API catalogue abbreviates this as `t`) | `>= 0` for aggregate operational emissions. Must be finite. A zero is plausible for a wholly zero-operational-emissions generation group, but unusual for total NSW generation. | Required generation/energy data or emissions factors were unavailable, or no aggregate result was returned. Missing must not be interpreted as zero emissions. |
| `market_value` | Calculated by OpenElectricity for each generator and interval as energy generated multiplied by the spot price in its network region, then summed. | `number \| null` / `float64` | AUD `$` | Any finite real number. Negative values are valid when electricity is generated during negative-price intervals. There is no useful fixed global bound because it depends on energy, price and interval length. | Energy, price, region mapping, or a required contributing observation was unavailable, so market value could not be calculated. |
| `storage_battery` | OpenElectricity battery state-of-charge series: total energy currently stored in batteries, reported for bidirectional battery units and aggregated by the network query. The catalogue uses average aggregation. | `number \| null` / `float64` | MWh | Normally `0 <= value <= total available battery storage capacity` for the selected region and date. The upper bound changes as facilities enter or leave service. | State-of-charge telemetry was not available, the period predates supported bidirectional battery reporting, or no eligible storage series was returned. Missing does **not** mean an empty battery. |

## Market metrics: price and demand

Endpoint: `GET /v4/market/network/NEM`, filtered to `network_region=NSW1`.

| Field | Source and derivation | API/Pandas type | Unit | Expected range / validation rule | Meaning of a missing value |
|---|---|---|---|---|---|
| `price` | AEMO NEM regional spot price supplied through OpenElectricity. Five-minute settlement has applied since October 2021; older trading arrangements differ. The API uses average aggregation for wider intervals. | `number \| null` / `float64` | AUD `$/MWh` | May be negative. Validate against the NEM market price floor and cap that applied at the observation date; these limits change over time, so one fixed bound should not be applied across the complete historical dataset. OpenElectricity's current guide quotes `-1,000` to `15,500` $/MWh, but historical and later limits must be checked by effective date. | No regional price observation was available or the timestamp lies outside price-data coverage. It does not mean a zero price. |
| `demand` | AEMO `DISPATCHREGIONSUM.TOTALDEMAND`: scheduled demand met by scheduled and semi-scheduled generation and interconnector flows. OpenElectricity uses average aggregation. | `number \| null` / `float64` | MW | Normally `>= 0`, finite, and within a plausible region-specific demand envelope. | The AEMO regional demand summary was unavailable for that interval or the interval is outside coverage. |
| `demand_energy` | OpenElectricity energy derived from demand over the interval. Energy metrics use time integration and are summed across wider intervals. | `number \| null` / `float64` | MWh | Normally `>= 0`. For an hourly series it should be broadly consistent with adjacent demand readings, allowing for trapezoidal integration. | Demand needed for the energy calculation was missing, or the derived energy series was unavailable for that historical period. |
| `demand_gross` | OpenElectricity gross demand: AEMO operational demand (`DEMAND_AND_NONSCHEDGEN`) plus AEMO rooftop-solar estimates. Rooftop data are half-hourly and interpolated to five-minute intervals before aggregation. | `number \| null` / `float64` | MW | Normally `>= 0`. It will generally be at least as large as scheduled demand, although source timing, definitions and aggregation can create exceptions that require investigation rather than automatic deletion. | Operational-demand data, rooftop-solar estimates, or their alignment was unavailable. Earlier periods can be structurally missing because rooftop estimates do not cover the full market history. |
| `demand_gross_energy` | Energy derived from gross demand over the interval and summed for wider intervals. | `number \| null` / `float64` | MWh | Normally `>= 0`; at `1h`, it should be broadly consistent with gross-demand power after allowing for integration. | Gross-demand inputs required for energy integration were unavailable, or the derived series was not available for that period. |

## Market metrics: renewable generation and proportions

| Field | Source and derivation | API/Pandas type | Unit | Expected range / validation rule | Meaning of a missing value |
|---|---|---|---|---|---|
| `generation_renewable` | OpenElectricity sum of renewable fuel technologies plus rooftop solar, battery discharge and non-hybrid pumped-hydro generation. Hybrid `hydro_and_storage` facilities such as TUMUT3 are excluded. | `number \| null` / `float64` | MW | Normally `>= 0`, finite and capacity-plausible. | One or more required generation aggregates or the separately added rooftop-solar series was unavailable, or the metric was not available for that period. |
| `generation_renewable_energy` | Energy derived from `generation_renewable` power readings using trapezoidal integration. | `number \| null` / `float64` | MWh | Normally `>= 0`; should be consistent with renewable power and interval duration. | Renewable power inputs required for integration were unavailable, or the derived series was unavailable. |
| `generation_renewable_with_storage` | `generation_renewable` plus generation from hybrid hydro-with-storage facilities. It is intended to represent renewable output including storage. | `number \| null` / `float64` | MW | Normally `>= 0` and normally `>= generation_renewable` at the same timestamp, subject to missing inputs and aggregation alignment. | Base renewable generation or hybrid hydro-storage generation was unavailable. |
| `generation_renewable_with_storage_energy` | Energy derived from `generation_renewable_with_storage` power readings using trapezoidal integration. | `number \| null` / `float64` | MWh | Normally `>= 0` and normally `>= generation_renewable_energy`, subject to interval alignment. | Required renewable-with-storage power observations were unavailable, or the derived series was unavailable. |
| `renewable_proportion` | Calculated as `generation_renewable / demand_gross * 100`. | `number \| null` / `float64` | `%` | Normally `>= 0`, but **not hard-capped at 100**. Regional renewable generation can exceed regional gross demand when electricity is exported or stored. Values over 100 must be retained unless other evidence shows an error. | The renewable numerator or gross-demand denominator was missing, or the denominator was zero/undefined. |
| `renewable_with_storage_proportion` | Calculated as `generation_renewable_with_storage / demand_gross * 100`. | `number \| null` / `float64` | `%` | Normally `>= 0`, but may validly exceed 100. It should normally be at least `renewable_proportion`, subject to missing data and rounding. | The storage-adjusted renewable numerator or gross-demand denominator was missing, or the denominator was zero/undefined. |

## Market metrics: curtailment

OpenElectricity currently supports curtailment only for the NEM. It is calculated
from AEMO dispatch data by comparing unconstrained intermittent generation
forecasts (`UIGF`) with dispatched targets (`CLEAREDMW`). OpenElectricity shifts
the source curtailment timestamp to align targets with actual generation.

| Field | Source and derivation | API/Pandas type | Unit | Expected range / validation rule | Meaning of a missing value |
|---|---|---|---|---|---|
| `curtailment` | Total calculated curtailed renewable power across wind and utility solar. The API uses average aggregation. | `number \| null` / `float64` | MW | Conceptually `>= 0`. Negative results can occur in the delivered series because this is a calculation from forecasts, dispatch targets, revisions and aggregation; flag them for review rather than automatically replacing them. | Required AEMO dispatch/UIGF inputs were unavailable, the period predates coverage, or curtailment is unsupported for the selected network. |
| `curtailment_energy` | Energy associated with total curtailment, summed over the requested interval. | `number \| null` / `float64` | MWh | Conceptually `>= 0`; negative calculated values should be flagged and investigated. | Total curtailment power inputs required for the energy calculation were unavailable. |
| `curtailment_solar_utility` | Calculated utility-scale solar power that could have been generated but was curtailed. | `number \| null` / `float64` | MW | Conceptually `>= 0`; negative calculated values are review flags, not automatically missing. | Required utility-solar forecast or dispatch-target observations were unavailable. |
| `curtailment_solar_utility_energy` | Energy associated with utility-scale solar curtailment over the interval. | `number \| null` / `float64` | MWh | Conceptually `>= 0`; investigate negative calculated values. | Solar curtailment inputs required for energy calculation were unavailable. |
| `curtailment_wind` | Calculated wind power that could have been generated but was curtailed. | `number \| null` / `float64` | MW | Conceptually `>= 0`; negative calculated values are review flags. | Required wind forecast or dispatch-target observations were unavailable. |
| `curtailment_wind_energy` | Energy associated with wind curtailment over the interval. | `number \| null` / `float64` | MWh | Conceptually `>= 0`; investigate negative calculated values. | Wind-curtailment inputs required for energy calculation were unavailable. |

## Market metrics: interconnector flows

The import and export metrics are separate non-negative directional magnitudes,
not one signed net-flow column. Their physical upper bounds depend on the set of
interconnectors, direction, constraints and date; therefore a single permanent
numeric maximum is inappropriate.

| Field | Source and derivation | API/Pandas type | Unit | Expected range / validation rule | Meaning of a missing value |
|---|---|---|---|---|---|
| `flow_imports` | OpenElectricity regional aggregation of power entering the selected region through NEM interconnectors. | `number \| null` / `float64` | MW | `>= 0`; zero means no imports during the interval. Validate high values against the directional interconnector capacity and constraints that applied at that date. | Interconnector observations/topology needed to calculate imports were unavailable, the period predates flow-series coverage, or the network has no supported interconnectors. |
| `flow_imports_energy` | Imported energy accumulated over the interval from import power flows. | `number \| null` / `float64` | MWh | `>= 0`; capacity and interval-duration dependent. | Import power inputs required for energy calculation were unavailable. |
| `flow_exports` | OpenElectricity regional aggregation of power leaving the selected region through NEM interconnectors. | `number \| null` / `float64` | MW | `>= 0`; zero means no exports during the interval. Validate high values against applicable directional capacity and constraints. | Interconnector observations/topology needed to calculate exports were unavailable, the period predates flow-series coverage, or the network has no supported interconnectors. |
| `flow_exports_energy` | Exported energy accumulated over the interval from export power flows. | `number \| null` / `float64` | MWh | `>= 0`; capacity and interval-duration dependent. | Export power inputs required for energy calculation were unavailable. |

## Recommended missing-value policy

1. Preserve missing values during raw-data storage.
2. Measure missingness by metric and year before choosing a modelling window.
3. Distinguish structural unavailability from isolated gaps.
4. Never use blanket `fillna(0)`: zero has a real physical or market meaning.
5. If short gaps are imputed for modelling, add an imputation flag and fit the
   imputation rule using training data only.
6. Treat an entirely absent API result differently from a returned data point
   whose value is explicitly `null`.

## Sources

- [OpenElectricity API overview](https://docs.openelectricity.org.au/api-reference/overview/)
- [OpenElectricity network-data endpoint](https://docs.openelectricity.org.au/api-reference/data/get-network-data/)
- [OpenElectricity market-data endpoint](https://docs.openelectricity.org.au/api-reference/market/get-network-data/)
- [OpenElectricity SDK metric type reference](https://docs.openelectricity.org.au/sdk/typescript/types/)
- [Power and energy calculation](https://docs.openelectricity.org.au/guides/energy/)
- [Price and market value](https://docs.openelectricity.org.au/guides/price/)
- [Demand and gross demand](https://docs.openelectricity.org.au/guides/demand/)
- [Renewable metrics](https://docs.openelectricity.org.au/guides/renewables/)
- [Curtailment](https://docs.openelectricity.org.au/guides/curtailment/)
- [Emissions](https://docs.openelectricity.org.au/guides/emissions/)
- [Battery state of charge](https://docs.openelectricity.org.au/guides/batteries/)
- [Interconnectors](https://docs.openelectricity.org.au/guides/interconnectors/)
- OpenElectricity v4 `/metrics` catalogue, retrieved directly from
  `https://api.openelectricity.org.au/v4/metrics` on 27 September 2026.

