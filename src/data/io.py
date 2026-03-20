from pathlib import Path

import pandas as pd

SUPPORTED_FILE_TYPES = {"csv", "parquet"}


def ensure_directory(path: str | Path) -> Path:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path : str | Path
        Directory path.

    Returns
    -------
    Path
        Path object of the ensured directory.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_dataset(
    path: str | Path,
    file_type: str = "csv",
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load a dataset from disk.

    Parameters
    ----------
    path : str | Path
        Path to the input dataset.
    file_type : str, default="csv"
        File format. Supported: "csv", "parquet".
    sep : str, default=","
        Separator used for CSV files.
    encoding : str, default="utf-8"
        Encoding used for CSV files.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the file type is not supported.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    file_type = file_type.lower()

    if file_type not in SUPPORTED_FILE_TYPES:
        raise ValueError(
            f"Unsupported file_type '{file_type}'. "
            f"Supported types: {SUPPORTED_FILE_TYPES}"
        )

    if file_type == "csv":
        return pd.read_csv(path, sep=sep, encoding=encoding)

    if file_type == "parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unhandled file_type: {file_type}")


def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    file_type: str = "csv",
    index: bool = False,
) -> None:
    """
    Save a DataFrame to disk.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str | Path
        Output file path.
    file_type : str, default="csv"
        File format. Supported: "csv", "parquet".
    index : bool, default=False
        Whether to save the index.

    Raises
    ------
    ValueError
        If the file type is not supported.
    """
    path = Path(path)
    ensure_directory(path.parent)

    file_type = file_type.lower()

    if file_type not in SUPPORTED_FILE_TYPES:
        raise ValueError(
            f"Unsupported file_type '{file_type}'. "
            f"Supported types: {SUPPORTED_FILE_TYPES}"
        )

    if file_type == "csv":
        df.to_csv(path, index=index)
        return

    if file_type == "parquet":
        df.to_parquet(path, index=index)
        return

    raise ValueError(f"Unhandled file_type: {file_type}")
