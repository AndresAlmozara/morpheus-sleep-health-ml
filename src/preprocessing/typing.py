import pandas as pd


def validate_columns_exist(df: pd.DataFrame, columns: list[str]) -> None:
    """
    Validate that all specified columns exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str]
        Columns that must exist in the DataFrame.

    Raises
    ------
    ValueError
        If one or more columns are missing.
    """
    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"The following columns are missing from the DataFrame: {missing_columns}"
        )


def get_target_column(config: dict) -> str:
    """
    Get the target column name from config.

    Parameters
    ----------
    config : dict
        Project configuration dictionary.

    Returns
    -------
    str
        Target column name.

    Raises
    ------
    ValueError
        If target is not defined in config.
    """
    target_col = config.get("target")

    if not target_col:
        raise ValueError("Target column is not defined in config under 'target'.")

    return target_col


def get_feature_groups(df: pd.DataFrame, config: dict) -> dict:
    """
    Get target, numeric features, and categorical features from config,
    validating that all declared columns exist and that target is not
    included as a feature.

    Expected config structure:
    {
        "target": "target_column_name",
        "drop_columns": [...],   # optional
        "features": {
            "numeric": [...],
            "categorical": [...]
        }
    }

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    config : dict
        Project configuration dictionary.

    Returns
    -------
    dict
        Dictionary with:
        - target: str
        - numeric_features: list[str]
        - categorical_features: list[str]

    Raises
    ------
    ValueError
        If target is included in feature lists, if duplicated features
        exist across groups, or if dropped columns are also declared as features.
    """
    target_col = get_target_column(config)

    features_cfg = config.get("features", {})
    numeric_features = features_cfg.get("numeric", [])
    categorical_features = features_cfg.get("categorical", [])
    drop_cols = config.get("drop_columns", [])

    validate_columns_exist(df, [target_col])
    validate_columns_exist(df, numeric_features)
    validate_columns_exist(df, categorical_features)

    all_features = numeric_features + categorical_features

    if target_col in all_features:
        raise ValueError(
            f"Target column '{target_col}' must not be included in feature lists."
        )

    duplicated_features = sorted(set(numeric_features) & set(categorical_features))
    if duplicated_features:
        raise ValueError(
            "The following columns are declared as both numeric and categorical: "
            f"{duplicated_features}"
        )

    dropped_feature_overlap = sorted(set(drop_cols) & set(all_features))
    if dropped_feature_overlap:
        raise ValueError(
            "The following columns appear in both 'drop_columns' and 'features': "
            f"{dropped_feature_overlap}"
        )

    return {
        "target": target_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
