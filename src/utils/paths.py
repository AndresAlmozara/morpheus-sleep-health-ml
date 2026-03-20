from pathlib import Path

# Project root directory
# This file lives in: src/utils/paths.py
# parents[2] goes up to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Main project folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CONFIG_DIR = PROJECT_ROOT / "configs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"


def get_raw_data_path(filename: str) -> Path:
    """
    Build the full path to a file inside data/raw.

    Parameters
    ----------
    filename : str
        File name, for example "heart.csv".

    Returns
    -------
    Path
        Full path inside data/raw.
    """
    return RAW_DATA_DIR / filename


def get_processed_data_path(filename: str) -> Path:
    """
    Build the full path to a file inside data/processed.

    Parameters
    ----------
    filename : str
        File name, for example "X_train_processed.csv".

    Returns
    -------
    Path
        Full path inside data/processed.
    """
    return PROCESSED_DATA_DIR / filename


def get_config_path(filename: str = "config.yaml") -> Path:
    """
    Build the full path to a config file inside configs.

    Parameters
    ----------
    filename : str, default="config.yaml"
        Config file name.

    Returns
    -------
    Path
        Full path inside configs.
    """
    return CONFIG_DIR / filename
