from __future__ import annotations

import numpy as np
import pandas as pd

SUPPORTED_OPERATORS = {">", ">=", "<", "<=", "==", "!="}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names:
    - strip leading/trailing spaces
    - lowercase
    - replace spaces with underscores

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized column names.
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing spaces from string-like columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with stripped string columns.
    """
    df = df.copy()
    string_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in string_cols:
        df[col] = df[col].str.strip()

    return df


def replace_empty_strings_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace empty or whitespace-only strings with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with empty strings replaced by NaN.
    """
    df = df.copy()
    string_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in string_cols:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    return df


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame without duplicate rows.
    """
    return df.drop_duplicates().copy()


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Drop selected columns if they exist.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str]
        Columns to drop.

    Returns
    -------
    pd.DataFrame
        DataFrame without selected columns.
    """
    df = df.copy()
    existing_columns = [col for col in columns if col in df.columns]
    return df.drop(columns=existing_columns)


def apply_row_filters(
    df: pd.DataFrame,
    filters: list[dict],
) -> pd.DataFrame:
    """
    Apply row filters defined as dictionaries with:
    - column
    - operator
    - value

    Supported operators:
    >, >=, <, <=, ==, !=

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    filters : list[dict]
        List of row filter rules.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.

    Raises
    ------
    ValueError
        If an operator is not supported.
    """
    df = df.copy()

    for rule in filters:
        column = rule["column"]
        operator = rule["operator"]
        value = rule["value"]

        if operator not in SUPPORTED_OPERATORS:
            raise ValueError(
                f"Unsupported operator '{operator}'. "
                f"Supported operators: {SUPPORTED_OPERATORS}"
            )

        if operator == ">":
            df = df[df[column] > value]
        elif operator == ">=":
            df = df[df[column] >= value]
        elif operator == "<":
            df = df[df[column] < value]
        elif operator == "<=":
            df = df[df[column] <= value]
        elif operator == "==":
            df = df[df[column] == value]
        elif operator == "!=":
            df = df[df[column] != value]

    return df


def run_basic_cleaning(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Run configurable basic cleaning steps.

    Expected config structure:
    {
        "cleaning": {
            "normalize_column_names": bool,
            "strip_strings": bool,
            "replace_empty_strings_with_nan": bool,
            "drop_duplicates": bool,
            "drop_columns": list[str],
            "row_filters": list[dict]
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
    pd.DataFrame
        Cleaned DataFrame.
    """
    cleaning_cfg = config.get("cleaning", {})

    if cleaning_cfg.get("normalize_column_names", False):
        df = normalize_column_names(df)

    if cleaning_cfg.get("strip_strings", False):
        df = strip_string_columns(df)

    if cleaning_cfg.get("replace_empty_strings_with_nan", False):
        df = replace_empty_strings_with_nan(df)

    if cleaning_cfg.get("drop_duplicates", False):
        df = drop_duplicate_rows(df)

    columns_to_drop = cleaning_cfg.get("drop_columns", [])
    if columns_to_drop:
        df = drop_columns(df, columns_to_drop)

    row_filters = cleaning_cfg.get("row_filters", [])
    if row_filters:
        df = apply_row_filters(df, row_filters)

    return df
