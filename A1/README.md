# Artifact 1 Artificial Intelligence Capstone Project

> [!IMPORTANT]
> Full execution of pipeline will take around 7 minutes to fetch entire dataset. Please be patient :)

This is the working repository for the A1 submission for Group P, COMP3742 Artificial Intelligence.

**Group Members:**

| **Name**       | **FAN**  |
|----------------|----------|
| Oliver Wuttke  | WUTT0019 |
| Hans Pujalte   | PUJA0009 |
| Shivansh Pant  | PANT0108 |
| Matilda Alford | ALFO0043 |

Each function contains an author in the comment stub which will contain the name and FAN of the group member who wrote it.

Please follow all steps in order to ensure pipeline runs smoothly :)

## Setup and Installation

Ensure python is installed on your system, version 3.11 or later. but <3.14

From the repository root, change into the [src](src) directory:

```bash
cd src
```

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

Then head to [Open Electricity Platform](https://platform.openelectricity.org.au/) and create an educational account by signing up with your flinders email.
You must do this to ensure you have access to historical data range.
Create your api key and then copy it and run the following command inside the [src](src) directory:

```bash
echo "OPENELECTRICITY_API_KEY=<PASTE YOUR API KEY HERE>" > .env
```

Make sure .env is UTF-8 or else python-dotenv will throw an error.

To run the pipline simply run:
```bash
python run_pipline.py
```

## Installing New Dependencies

Please make sure to update the [requirements.txt](src/requirements.txt) after installing new packages via:

```bash
pip freeze > requirements.txt
```


## Run from the repository root

Using the existing root virtual environment:

```bash
.venv/bin/python A1/src/run_pipeline.py
.venv/bin/python A1/scripts/check_energy_scale.py
```

The second command fetches only two sample days for comparison; it does not replace raw data. If using the `A1/src/.venv` environment created above, substitute its Python executable, or activate it and use `python`.

- Pipeline code: `src/`
- Cleaning notebook: [notebooks/data_cleaning.ipynb](notebooks/data_cleaning.ipynb)
- Raw and cleaned datasets: `data/raw/` and `data/processed/`
- API comparison evidence: `data/validation/`
- Cleaning report: `reports/cleaning_report.json` (written by the pipeline)
- Notebook charts: `reports/figures/histograms/` and `reports/figures/boxplots/`
- [Cleaning decisions](docs/cleaning_decisions.md)
- [Data dictionary](docs/data_dictionary.md)

Dependencies remain in `src/requirements.txt`, and the API key remains in `src/.env`.
The notebook resolves paths when launched from the repository root or within A1.
