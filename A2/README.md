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

Please follow all steps in order to ensure pipeline runs smoothly :)

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
echo "OPENELECTRICITY_API_KEY=<PASTE YOUR API KEY HERE>" > .env
```

Make sure .env is UTF-8 or else python-dotenv will throw an error.

## Training Pipeline
First step is retrieving the training dataset by running:
```bash
python A1_data_pipeline/run_pipeline
```

Now we can run our training pipeline:
```bash
python training_pipeline.py --input data/processed/electricity_transformed.parquet
```
This will save the model params and NEEDS to be done before inference.

## Inference
Ensure the above Training Pipeline is executed without error or this next part won't work.
Simply run the following command to perform inference on the latest instance:
```bash
python inference.py
```



