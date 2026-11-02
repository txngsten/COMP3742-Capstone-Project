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
If a file has only 1 author then functions/methods will not contain author notes

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
pip install -r requirements.txt
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
included in `requirements.txt`. Use the existing project environment if these
packages are already installed.

## Training pipeline

Generate A2's transformed dataset:

```bash
python A1_data_pipeline/run_pipeline.py
```

Train both implemented subsystems:

```bash
python training_pipeline.py --input A2/data/processed/electricity_transformed.parquet
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
python inference.py
```

The script always runs demand and network-stress inference in sequence on the same
history. It does not accept model-selection or historical-data arguments. For a
historical network-stress demonstration, use `notebooks/inference.ipynb`.

Live inference uses `OPENELECTRICITY_API_KEY` from the environment or `.env`;
an existing environment value takes precedence. Model and Prolog paths are resolved
relative to A2, independently of the launch directory. Train both models before
running inference.