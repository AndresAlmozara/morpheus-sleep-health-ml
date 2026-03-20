from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """
    Load a YAML configuration file and return it as a dictionary.

    Parameters
    ----------
    path : str | Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the YAML content is empty or invalid.
    """
    config_path = Path(path)

    # Check that the file exists before trying to open it
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Open and parse the YAML file
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # safe_load can return None if the YAML file is empty
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # We expect the config to be a dictionary at the top level
    if not isinstance(config, dict):
        raise ValueError("Config file must define a YAML dictionary/object at top level.")

    return config


def validate_config(config: dict) -> None:
    """
    Validate that the config contains the minimum required sections
    for the reusable tabular preprocessing system.

    Expected minimum structure:
    {
        "data": {...},
        "target": "...",
        "features": {
            "numeric": [...],
            "categorical": [...]
        },
        "preprocessing": {
            "numeric": {...},
            "categorical": {...}
        }
    }

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Raises
    ------
    ValueError
        If required sections or keys are missing.
    """
    required_top_level_keys = ["data", "target", "features", "preprocessing"]

    missing_top_level_keys = [
        key for key in required_top_level_keys if key not in config
    ]

    if missing_top_level_keys:
        raise ValueError(
            f"Missing required top-level config keys: {missing_top_level_keys}"
        )

    # Validate data section
    data_cfg = config["data"]
    if not isinstance(data_cfg, dict):
        raise ValueError("'data' section must be a dictionary.")

    required_data_keys = ["input_path", "file_type"]
    missing_data_keys = [key for key in required_data_keys if key not in data_cfg]

    if missing_data_keys:
        raise ValueError(
            f"Missing required keys in 'data' section: {missing_data_keys}"
        )

    # Validate target
    if not isinstance(config["target"], str) or not config["target"].strip():
        raise ValueError("'target' must be a non-empty string.")

    # Validate features section
    features_cfg = config["features"]
    if not isinstance(features_cfg, dict):
        raise ValueError("'features' section must be a dictionary.")

    required_feature_keys = ["numeric", "categorical"]
    missing_feature_keys = [
        key for key in required_feature_keys if key not in features_cfg
    ]

    if missing_feature_keys:
        raise ValueError(
            f"Missing required keys in 'features' section: {missing_feature_keys}"
        )

    if not isinstance(features_cfg["numeric"], list):
        raise ValueError("'features.numeric' must be a list.")

    if not isinstance(features_cfg["categorical"], list):
        raise ValueError("'features.categorical' must be a list.")

    # Validate preprocessing section
    preprocessing_cfg = config["preprocessing"]
    if not isinstance(preprocessing_cfg, dict):
        raise ValueError("'preprocessing' section must be a dictionary.")

    required_preprocessing_keys = ["numeric", "categorical"]
    missing_preprocessing_keys = [
        key for key in required_preprocessing_keys if key not in preprocessing_cfg
    ]

    if missing_preprocessing_keys:
        raise ValueError(
            f"Missing required keys in 'preprocessing' section: "
            f"{missing_preprocessing_keys}"
        )

    numeric_cfg = preprocessing_cfg["numeric"]
    categorical_cfg = preprocessing_cfg["categorical"]

    if not isinstance(numeric_cfg, dict):
        raise ValueError("'preprocessing.numeric' must be a dictionary.")

    if not isinstance(categorical_cfg, dict):
        raise ValueError("'preprocessing.categorical' must be a dictionary.")


def load_and_validate_config(path: str | Path) -> dict:
    """
    Load a YAML config file and validate its minimum structure.

    Parameters
    ----------
    path : str | Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Loaded and validated configuration dictionary.
    """
    config = load_config(path)
    validate_config(config)
    return config
