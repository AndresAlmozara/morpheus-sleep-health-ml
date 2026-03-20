from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def make_numeric_imputer(strategy: str = "median") -> SimpleImputer:
    """
    Create a numeric imputer.

    Parameters
    ----------
    strategy : str, default="median"
        Supported strategies: "mean", "median", "most_frequent", "constant"

    Returns
    -------
    SimpleImputer
        Configured numeric imputer.

    Raises
    ------
    ValueError
        If the strategy is not supported.
    """
    supported_strategies = {"mean", "median", "most_frequent", "constant"}

    if strategy not in supported_strategies:
        raise ValueError(
            f"Unsupported numeric imputation strategy '{strategy}'. "
            f"Supported: {supported_strategies}"
        )

    # If constant is selected for numeric columns,
    # we use 0 as a simple default fill value in this v1.
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=0)

    return SimpleImputer(strategy=strategy)


def make_categorical_imputer(
    strategy: str = "most_frequent",
    fill_value: str = "missing",
) -> SimpleImputer:
    """
    Create a categorical imputer.

    Parameters
    ----------
    strategy : str, default="most_frequent"
        Supported strategies: "most_frequent", "constant"
    fill_value : str, default="missing"
        Value used when strategy is "constant".

    Returns
    -------
    SimpleImputer
        Configured categorical imputer.

    Raises
    ------
    ValueError
        If the strategy is not supported.
    """
    supported_strategies = {"most_frequent", "constant"}

    if strategy not in supported_strategies:
        raise ValueError(
            f"Unsupported categorical imputation strategy '{strategy}'. "
            f"Supported: {supported_strategies}"
        )

    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=fill_value)

    return SimpleImputer(strategy=strategy)


def make_scaler(name: str = "standard"):
    """
    Create a scaler for numeric features.

    Parameters
    ----------
    name : str, default="standard"
        Supported scalers: "standard", "minmax", "none"

    Returns
    -------
    transformer
        Sklearn scaler object or "passthrough" if no scaling is requested.

    Raises
    ------
    ValueError
        If the scaler name is not supported.
    """
    supported_scalers = {"standard", "minmax", "none"}

    if name not in supported_scalers:
        raise ValueError(
            f"Unsupported scaler '{name}'. Supported: {supported_scalers}"
        )

    if name == "standard":
        return StandardScaler()

    if name == "minmax":
        return MinMaxScaler()

    return "passthrough"


def make_encoder(name: str = "onehot"):
    """
    Create an encoder for categorical features.

    Parameters
    ----------
    name : str, default="onehot"
        Supported encoders: "onehot"

    Returns
    -------
    transformer
        Configured categorical encoder.

    Raises
    ------
    ValueError
        If the encoder name is not supported.
    """
    supported_encoders = {"onehot"}

    if name not in supported_encoders:
        raise ValueError(
            f"Unsupported encoder '{name}'. Supported: {supported_encoders}"
        )

    # handle_unknown="ignore" avoids errors when test data contains
    # categories not seen during fit on train.
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_numeric_pipeline(
    imputer_strategy: str = "median",
    scaler_name: str = "standard",
) -> Pipeline:
    """
    Build a numeric preprocessing pipeline:
    imputation -> scaling

    Parameters
    ----------
    imputer_strategy : str, default="median"
        Strategy used for numeric missing values.
    scaler_name : str, default="standard"
        Scaling method for numeric features.

    Returns
    -------
    Pipeline
        Sklearn pipeline for numeric columns.
    """
    return Pipeline(
        steps=[
            ("imputer", make_numeric_imputer(strategy=imputer_strategy)),
            ("scaler", make_scaler(name=scaler_name)),
        ]
    )


def build_categorical_pipeline(
    imputer_strategy: str = "most_frequent",
    encoder_name: str = "onehot",
    fill_value: str = "missing",
) -> Pipeline:
    """
    Build a categorical preprocessing pipeline:
    imputation -> encoding

    Parameters
    ----------
    imputer_strategy : str, default="most_frequent"
        Strategy used for categorical missing values.
    encoder_name : str, default="onehot"
        Encoding method for categorical features.
    fill_value : str, default="missing"
        Value used when categorical imputer strategy is "constant".

    Returns
    -------
    Pipeline
        Sklearn pipeline for categorical columns.
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                make_categorical_imputer(
                    strategy=imputer_strategy,
                    fill_value=fill_value,
                ),
            ),
            ("encoder", make_encoder(name=encoder_name)),
        ]
    )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict,
) -> ColumnTransformer:
    """
    Build the final ColumnTransformer using numeric and categorical feature groups.

    Expected config structure:
    {
        "preprocessing": {
            "numeric": {
                "imputer_strategy": "median",
                "scaler": "standard"
            },
            "categorical": {
                "imputer_strategy": "most_frequent",
                "encoder": "onehot",
                "fill_value": "missing"
            }
        }
    }

    Parameters
    ----------
    numeric_features : list[str]
        List of numeric feature names.
    categorical_features : list[str]
        List of categorical feature names.
    config : dict
        Project configuration dictionary.

    Returns
    -------
    ColumnTransformer
        Final sklearn preprocessor for the dataset.
    """
    preprocessing_cfg = config.get("preprocessing", {})
    numeric_cfg = preprocessing_cfg.get("numeric", {})
    categorical_cfg = preprocessing_cfg.get("categorical", {})

    # Build the numeric block only if numeric features exist
    numeric_pipeline = build_numeric_pipeline(
        imputer_strategy=numeric_cfg.get("imputer_strategy", "median"),
        scaler_name=numeric_cfg.get("scaler", "standard"),
    )

    # Build the categorical block only if categorical features exist
    categorical_pipeline = build_categorical_pipeline(
        imputer_strategy=categorical_cfg.get("imputer_strategy", "most_frequent"),
        encoder_name=categorical_cfg.get("encoder", "onehot"),
        fill_value=categorical_cfg.get("fill_value", "missing"),
    )

    transformers = []

    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
