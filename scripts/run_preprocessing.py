import sys
from pathlib import Path

import pandas as pd

from src.data.io import load_dataset, save_dataframe
from src.data.split import make_train_test_split, split_features_target
from src.preprocessing.cleaning import run_basic_cleaning
from src.preprocessing.pipeline_builder import build_preprocessor
from src.preprocessing.typing import get_feature_groups
from src.utils.config import load_and_validate_config
from src.utils.paths import get_config_path, get_processed_data_path


def to_dataframe(transformed_data, feature_names: list[str] | None = None) -> pd.DataFrame:
    """
    Convert transformed sklearn output into a pandas DataFrame.

    Parameters
    ----------
    transformed_data : array-like
        Output of sklearn transform / fit_transform.
    feature_names : list[str] | None, default=None
        Column names for the resulting DataFrame.

    Returns
    -------
    pd.DataFrame
        Transformed data as DataFrame.
    """
    return pd.DataFrame(transformed_data, columns=feature_names)


def get_split_config(config: dict) -> dict:
    """
    Read split configuration with sensible defaults.

    Expected optional structure:
    {
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "stratify": true
        }
    }

    Parameters
    ----------
    config : dict
        Project configuration dictionary.

    Returns
    -------
    dict
        Split parameters.
    """
    split_cfg = config.get("split", {})

    return {
        "test_size": split_cfg.get("test_size", 0.2),
        "random_state": split_cfg.get("random_state", 42),
        "stratify": split_cfg.get("stratify", False),
    }


def get_save_config(config: dict) -> dict:
    """
    Read save/output configuration with defaults.

    Expected optional structure:
    {
        "save": {
            "save_cleaned_data": false,
            "cleaned_data_filename": "cleaned_data.csv",
            "x_train_filename": "X_train_processed.csv",
            "x_test_filename": "X_test_processed.csv",
            "y_train_filename": "y_train.csv",
            "y_test_filename": "y_test.csv"
        }
    }

    Parameters
    ----------
    config : dict
        Project configuration dictionary.

    Returns
    -------
    dict
        Save parameters.
    """
    save_cfg = config.get("save", {})

    return {
        "save_cleaned_data": save_cfg.get("save_cleaned_data", False),
        "cleaned_data_filename": save_cfg.get("cleaned_data_filename", "cleaned_data.csv"),
        "x_train_filename": save_cfg.get("x_train_filename", "X_train_processed.csv"),
        "x_test_filename": save_cfg.get("x_test_filename", "X_test_processed.csv"),
        "y_train_filename": save_cfg.get("y_train_filename", "y_train.csv"),
        "y_test_filename": save_cfg.get("y_test_filename", "y_test.csv"),
    }


def run_preprocessing(config_path: str | Path | None = None) -> None:
    """
    Run the full preprocessing workflow:

    1. Load config
    2. Load raw dataset
    3. Apply safe table cleaning
    4. Read target / numeric / categorical feature groups
    5. Split X/y and train/test
    6. Build sklearn preprocessor
    7. Fit on X_train and transform train/test
    8. Save final processed outputs

    Parameters
    ----------
    config_path : str | Path | None, default=None
        Path to YAML config file. If None, uses configs/config.yaml.
    """
    # ------------------------------------------------------------------
    # 1. Resolve config path
    # ------------------------------------------------------------------
    if config_path is None:
        config_path = get_config_path()

    # ------------------------------------------------------------------
    # 2. Load and validate config
    # ------------------------------------------------------------------
    config = load_and_validate_config(config_path)

    # ------------------------------------------------------------------
    # 3. Read dataset loading parameters
    # ------------------------------------------------------------------
    data_cfg = config["data"]
    input_path = data_cfg["input_path"]
    file_type = data_cfg["file_type"]
    sep = data_cfg.get("sep", ",")
    encoding = data_cfg.get("encoding", "utf-8")

    # ------------------------------------------------------------------
    # 4. Load raw dataset
    # ------------------------------------------------------------------
    df_raw = load_dataset(
        path=input_path,
        file_type=file_type,
        sep=sep,
        encoding=encoding,
    )

    # ------------------------------------------------------------------
    # 5. Apply safe cleaning before split
    # ------------------------------------------------------------------
    df_clean = run_basic_cleaning(df_raw, config)

    # ------------------------------------------------------------------
    # 6. Optionally save cleaned data for inspection/debugging
    # ------------------------------------------------------------------
    save_cfg = get_save_config(config)

    if save_cfg["save_cleaned_data"]:
        save_dataframe(
            df_clean,
            get_processed_data_path(save_cfg["cleaned_data_filename"]),
            file_type="csv",
            index=False,
        )

    # ------------------------------------------------------------------
    # 7. Get target / numeric / categorical groups
    # ------------------------------------------------------------------
    feature_info = get_feature_groups(df_clean, config)

    target_col = feature_info["target"]
    numeric_features = feature_info["numeric_features"]
    categorical_features = feature_info["categorical_features"]

    # ------------------------------------------------------------------
    # 8. Split features and target
    # ------------------------------------------------------------------
    X, y = split_features_target(df_clean, target_col=target_col)

    # ------------------------------------------------------------------
    # 9. Train/test split
    # ------------------------------------------------------------------
    split_cfg = get_split_config(config)

    X_train, X_test, y_train, y_test = make_train_test_split(
        X=X,
        y=y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=split_cfg["stratify"],
    )

    # ------------------------------------------------------------------
    # 10. Build final sklearn preprocessor
    # ------------------------------------------------------------------
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        config=config,
    )

    # ------------------------------------------------------------------
    # 11. Fit on train and transform train/test
    # ------------------------------------------------------------------
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # ------------------------------------------------------------------
    # 12. Recover transformed feature names from sklearn
    # ------------------------------------------------------------------
    feature_names = preprocessor.get_feature_names_out()

    # ------------------------------------------------------------------
    # 13. Convert transformed arrays into DataFrames
    # ------------------------------------------------------------------
    X_train_processed_df = to_dataframe(
        transformed_data=X_train_processed,
        feature_names=feature_names.tolist(),
    )

    X_test_processed_df = to_dataframe(
        transformed_data=X_test_processed,
        feature_names=feature_names.tolist(),
    )

    # ------------------------------------------------------------------
    # 14. Save processed outputs
    # ------------------------------------------------------------------
    save_dataframe(
        X_train_processed_df,
        get_processed_data_path(save_cfg["x_train_filename"]),
        file_type="csv",
        index=False,
    )

    save_dataframe(
        X_test_processed_df,
        get_processed_data_path(save_cfg["x_test_filename"]),
        file_type="csv",
        index=False,
    )

    save_dataframe(
        y_train.to_frame(name=target_col),
        get_processed_data_path(save_cfg["y_train_filename"]),
        file_type="csv",
        index=False,
    )

    save_dataframe(
        y_test.to_frame(name=target_col),
        get_processed_data_path(save_cfg["y_test_filename"]),
        file_type="csv",
        index=False,
    )

    print("Preprocessing completed successfully.")
    print(f"Saved: {save_cfg['x_train_filename']}")
    print(f"Saved: {save_cfg['x_test_filename']}")
    print(f"Saved: {save_cfg['y_train_filename']}")
    print(f"Saved: {save_cfg['y_test_filename']}")


if __name__ == "__main__":
    # Allow optional custom config path from command line:
    # python scripts/run_preprocessing.py configs/config.yaml
    custom_config_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_preprocessing(custom_config_path)
