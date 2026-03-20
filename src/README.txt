# src/

Reusable core code for tabular ML projects.

This folder contains the project logic that should be importable, modular, and reusable.
It is the engine of the workflow.

## Philosophy

- notebooks = EDA, reasoning, interpretation
- config = declared decisions
- src = systematic execution
- scripts = orchestration entry points

---

## Structure

```text
src/
├── data/
│   ├── io.py
│   └── split.py
│
├── preprocessing/
│   ├── cleaning.py
│   ├── typing.py
│   └── pipeline_builder.py
│
└── utils/
    ├── config.py
    └── paths.py
```

---

## `data/`

Utilities for reading, saving, and splitting datasets.

- `io.py` -> dataset input/output
- `split.py` -> feature/target split and train/test split

### `src/data/io.py`
Handles dataset loading and saving.

**Responsibilities**
- load datasets from disk
- save processed outputs
- keep file I/O separate from preprocessing logic

**Does not handle**
- cleaning
- feature typing
- preprocessing
- model logic

### `src/data/split.py`
Handles dataset partitioning.

**Responsibilities**
- separate `X` and `y`
- create reproducible train/test splits
- optionally apply stratification

**Does not handle**
- loading/saving
- cleaning
- encoding or scaling
- feature engineering

**Typical flow**
`df_clean -> X, y -> X_train, X_test, y_train, y_test`

---

## `preprocessing/`

Reusable preprocessing logic for tabular data.

- `cleaning.py` -> safe table cleaning before split
- `typing.py` -> target and feature group definition
- `pipeline_builder.py` -> final sklearn preprocessor builder

### `src/preprocessing/cleaning.py`
Safe, declarative cleaning before split and sklearn preprocessing.

**Responsibilities**
- normalize column names
- strip string values
- replace empty strings with missing values
- drop duplicates
- drop selected columns
- apply explicit row filters

**Does not handle**
- statistical imputation
- scaling
- encoding
- fit-on-train transformations
- model-specific logic

**Typical flow**
`df_raw -> df_clean`

### `src/preprocessing/typing.py`
Defines the logical role of columns.

**Responsibilities**
- read target from config
- read numeric and categorical feature groups
- validate that declared columns exist
- ensure target is not included in feature lists

**Does not handle**
- automatic feature discovery
- cleaning
- split
- preprocessing execution

**Typical flow**
`df_clean -> target + feature groups`

### `src/preprocessing/pipeline_builder.py`
Builds the final sklearn preprocessing object.

**Responsibilities**
- create imputers, scalers, and encoders
- build numeric and categorical pipelines
- assemble the final `ColumnTransformer`

**Does not handle**
- raw data loading
- cleaning
- train/test split
- model fitting
- business or analytical decisions

**Typical flow**
`feature groups + preprocessing config -> ColumnTransformer`

---

## `utils/`

Shared helper utilities.

- `config.py` -> config loading and validation
- `paths.py` -> project path definitions and path helpers

### `src/utils/config.py`
Loads and validates the project YAML config.

**Responsibilities**
- load YAML into a Python dictionary
- validate the minimum required structure
- provide a single config entry point for scripts and modules

**Does not handle**
- data loading
- preprocessing
- modeling

**Typical flow**
`config.yaml -> load_and_validate_config(...) -> config dict`

### `src/utils/paths.py`
Centralizes project paths.

**Responsibilities**
- define root, data, config, and output directories
- provide simple helpers for common file paths
- reduce hardcoded paths across the project

**Does not handle**
- file reading/writing
- config validation
- preprocessing logic

**Typical flow**
`paths -> load/save operations`

---

## Workflow role of `src/`

`src/` provides reusable building blocks.
These modules can be imported independently in notebooks or scripts.

Typical end-to-end flow:

`config -> raw data -> cleaning -> feature roles -> split -> preprocessor`

The high-level orchestration lives outside `src/`, usually in `scripts/`.
