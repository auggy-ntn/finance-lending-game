# Preprocessing for model training
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import constants.constants as cst
from src.utils.model_utils import save_model


def create_preprocessor() -> ColumnTransformer:
    """Create a preprocessor for the model.

    Returns:
        ColumnTransformer: The preprocessor.
    """
    # Create a copy of the list to avoid modifying the original
    numeric_features = [
        feat for feat in cst.NUMERICAL if feat not in [cst.SIGNAL_1, cst.SIGNAL_3]
    ]
    categorical_features = cst.CATEGORICAL

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        drop="first", handle_unknown="ignore", sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def preprocess_data(
    data: pd.DataFrame,
    preprocessor: Optional[ColumnTransformer] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, pd.Series | None, ColumnTransformer]:
    """Preprocess the data for model training or prediction.

    Args:
        data (pd.DataFrame): The input data.
        preprocessor (Optional[ColumnTransformer]): The preprocessor to use.
            If None, creates and fits a new one.
        fit (bool): Whether to fit the preprocessor
            (True for training, False for prediction).

    Returns:
        pd.DataFrame: The preprocessed data.
        pd.Series: The target variable (None if not in data).
        ColumnTransformer: The fitted preprocessor.
    """

    if preprocessor is None:
        logger.info("No preprocessor provided. Creating a new one.")
        preprocessor = create_preprocessor()

    # Apply log transformation to income to handle skewness
    if cst.INCOME in data.columns:
        logger.info(f"Applying log transformation to {cst.INCOME}")
        data[cst.INCOME] = np.log1p(data[cst.INCOME])

    y = None
    if cst.TARGET in data.columns:
        y = data[cst.TARGET]
        data = data.drop(columns=[cst.TARGET])

    # Fit and transform for training, only transform for prediction
    if fit:
        logger.info("Fitting and transforming data.")
        data_preprocessed_array = preprocessor.fit_transform(data)
        save_model(preprocessor, "preprocessor")
        logger.info("Preprocessor fitted and saved.")
    else:
        logger.info("Transforming data with existing preprocessor.")
        data_preprocessed_array = preprocessor.transform(data)

    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "cat":
            feature_names.extend(transformer.get_feature_names_out(columns))
        elif name == "num":
            feature_names.extend(columns)
        elif name != "remainder":
            feature_names.extend(columns)

    data_preprocessed = pd.DataFrame(
        data_preprocessed_array,
        columns=feature_names,
    )

    # Return X, y, and the preprocessor
    return data_preprocessed, y, preprocessor
