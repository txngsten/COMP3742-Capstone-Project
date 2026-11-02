# Artifact 2 Artificial Intelligence Capstone Project

This is the working repository for the A2 submission for Group P, COMP3742 Artificial Intelligence.

**Group Members:**

| **Name**       | **FAN**  |
|----------------|----------|
| Oliver Wuttke  | WUTT0019 |
| Hans Pujalte   | PUJA0009 |
| Shivansh Pant  | PANT0108 |
| Matilda Alford | ALFO0043 |

Each function contains an author in the comment stub which will contain the name and FAN of the group member who wrote it.
If a file has only 1 author then functions/methods will not contain author notes.

Run the following commands from the repository root. From within A2, omit the `A2/` prefix.

## Setup and Installation

Ensure python is installed on your system, version 3.11 or later. but <3.14
Create and activate your virtual environment.

```bash
python -m venv .venv          # create
source .venv/bin/activate     # activate (Linux/macOS)
.venv\Scripts\activate        # activate (Windows)
```

Install the dependencies using pip:

```bash
pip install -r A2/requirements.txt
```

> [!IMPORTANT]
> If you are marking the env file will be already installed you can skip this step.

Then head to [Open Electricity Platform](https://platform.openelectricity.org.au/) and create an educational account by signing up with your flinders email.
You must do this to ensure you have access to historical data range.
Create your api key and then copy it and run the following command :

```bash
echo "OPENELECTRICITY_API_KEY=<PASTE YOUR API KEY HERE>" > A2/.env
```

Make sure .env is UTF-8 or else python-dotenv will throw an error.

## Runtime requirements

Install SWI-Prolog separately so PySwip can locate its shared library. XGBoost is
included in `A2/requirements.txt`. Use the existing project environment if these
packages are already installed.

## Training pipeline

Generate A2's transformed dataset:

```bash
python A2/A1_data_pipeline/run_pipeline.py
```

Train both implemented subsystems (demand and network stress):

```bash
python A2/training_pipeline.py --input A2/data/processed/electricity_transformed.parquet
```

The `--input` argument is required and specifies the transformed dataset to load.
Paths are relative to the directory where the command is run; from A2, use
`--input data/processed/electricity_transformed.parquet`. The script loads the dataset
once and trains demand and network stress in sequence. There is no model-selection
argument. Generate the dataset first, or provide an existing transformed parquet.

The demand model's existing training and plotting behaviour is preserved. Importing
either entry-point module does not start training or fetch data.

## Inference

Run both subsystems against recent transformed observations fetched once from the API:

```bash
python A2/inference.py
```

The script always runs demand and network-stress inference in sequence on the same
history. It does not accept model-selection or historical-data arguments. For a
historical network-stress demonstration, use `notebooks/inference.ipynb`.

Live inference uses `OPENELECTRICITY_API_KEY` from the environment or `A2/.env`;
an existing environment value takes precedence. Model and Prolog paths are resolved
relative to A2, independently of the launch directory. Train both models before
running inference.

## Network-stress method

`models/NetworkStressModel.py` implements `train_eval(training_set)`, the compatible
`train(training_set)` alias, and `inference(instance)`. Training accepts a timezone-aware
`timestamp` column or a DatetimeIndex. Inference expects sorted, unique, timezone-aware
DatetimeIndex history and predicts from its latest row. The shared script uses the latest fetched observation; the notebook can select a
historical cutoff. Both display the returned prediction, potential stress drivers,
and advisory recommendations.

- The target is a one-hour-ahead regional stress proxy with low, medium, high and
  critical classes. It is not a direct engineering measurement of network stress.
- Adequacy combines gross demand (0.40), price (0.35), and imports (0.25).
  Congestion combines nonnegative curtailment (0.60), exports (0.25), and renewable
  proportion including storage (0.15). The larger score determines the class.
- Severity thresholds are the training period's 75th, 90th and 97th percentiles.
  Chronological 60/20/20 boundaries are assigned by the next-hour target timestamp.
- Rows with imputed current or next-hour measurements, and nonconsecutive hourly
  pairs, are excluded. Lagged and rolling inputs can still contain previously
  interpolated measurements; this is a limitation of the inherited preprocessing.
- Numeric predictors include the engineered time and historical features. The
  timestamp, imputation indicator, target fields and A1 cluster/anomaly labels are
  excluded. Missing/infinite required inputs produce an explicit error.
- Random Forest and XGBoost use the branch's class-imbalance handling and fixed
  settings. Validation macro-F1 selects the algorithm. It is refitted on training
  plus validation, with thresholds kept fixed, before the test evaluation.
- The latest branch's simplified Prolog rules identify possible adequacy/congestion
  drivers and recommend review actions. They do not change the ML risk class or
  accept reserve-margin overrides. Facts are cleared between calls. No recommendation
  sends a notification or performs an operational action.

## Files and saved outputs

| Location | Purpose |
| --- | --- |
| `models/NetworkStressModel.py` | Network-stress training, evaluation and inference |
| `prolog/stress_rules.pl` | Network-stress symbolic rules |
| `notebooks/model_training.ipynb` | Training and evaluation demonstration |
| `notebooks/inference.ipynb` | Historical hybrid inference demonstration |
| `models/network_stress_model.joblib` | Selected estimator, ordered feature list, class mapping and target thresholds |
| `models/network_stress_model_metadata.json` | Matching training configuration, split ranges, metrics and confusion matrix |
| `demand_gp.npz` | Existing demand-model parameters |

Run training to create or refresh the network-stress bundle and metadata together.
The older standalone XGBoost JSON export is not required by inference and is not
copied from the feature branch. Only load trusted model bundles.

The notebooks call the integrated class rather than maintaining a second training
implementation. Set their `DATA_PATH` to an existing transformed parquet if A2's
pipeline has not yet generated its default input. The price and renewable model
placeholders are not invoked by the shared entry points.
