# Artifact 1 Artificial Intelligence Capstone Project
This is the working repository for the A1 submission for Group P, COMP3742 Artificial Intelligence.

**Group Members:**

| **Name**          | **FAN**      |
|---------------|----------|
| Oliver Wuttke | WUTT0019 |
|               |          |
|               |          |
|               |          |

Each function contains an author in the comment stub which will contain the name and FAN of the group member who wrote it.

Please follow all steps in order to ensure pipeline runs smoothly :)

## Setup and Installation
Ensure python is installed on your system, version 3.11 or later.

Change into the [src](src) directory:
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

## Installing New Dependencies
Please make sure to update the [requirements.txt](src/requirements.txt) after installing new packages via:
```bash
pip freeze > requirements.txt
```
