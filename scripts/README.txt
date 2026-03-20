# scripts/

This folder contains executable orchestration scripts.

These scripts are **entry points** for complete workflows.  
They are not the reusable core of the project.

The reusable logic lives in `src/`.  
Scripts in this folder call that logic to execute end-to-end tasks.

---

## `run_preprocessing.py`

Main orchestration script for the reusable tabular preprocessing workflow.

### What it does
This script runs the full preprocessing pipeline from configuration to saved outputs.

### Main responsibilities
- load and validate the YAML config
- load the raw dataset
- apply safe table cleaning
- read target, numeric features, and categorical features
- split the dataset into train and test sets
- build the sklearn preprocessing object
- fit the preprocessor on training data
- transform training and test data
- save final processed outputs

### Expected outputs
By default, the script saves:

- `data/processed/X_train_processed.csv`
- `data/processed/X_test_processed.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`

Optional output:

- `data/processed/cleaned_data.csv`

### When to use it
Use this script when the project has a **single clear preprocessing recipe** and you want a reproducible processed dataset saved to disk.

Typical use cases:
- one main modeling workflow
- one dominant preprocessing strategy
- portfolio projects with a stable preprocessing pipeline
- situations where saving processed outputs is useful

### When not to use it as the main workflow
In multi-model experiments, where different models require different preprocessing strategies, it is often better to:

- import reusable functions directly from `src/`
- build different preprocessors in the modeling notebook or script
- keep preprocessing inside each sklearn `Pipeline`

In those cases, `run_preprocessing.py` is still valid as a reference workflow, but it may not be the main execution path.

### Typical usage
python scripts/run_preprocessing.py

### Custom config path
You can also provide a custom config file:

python scripts/run_preprocessing.py configs/config.yaml

### Workflow summary
The script follows this sequence:

config -> raw data -> cleaning -> feature roles -> split -> preprocessor -> fit on train -> transform train/test -> save outputs