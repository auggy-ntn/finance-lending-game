# Preprocessing for model training
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import constants.constants as cst


def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series | None]:
    """Preprocess the data for model training.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The preprocessed data.
        pd.Series: The target variable.
    """

    y = None
    # Separate target variable if it exists
    if cst.TARGET in data.columns:
        y = data[cst.TARGET]
        data = data.drop(columns=[cst.TARGET])

    # One-hot encode categorical variables
    ohe = OneHotEncoder(sparse_output=False, drop="first")
    cat_data = ohe.fit_transform(data[cst.CATEGORICAL])
    cat_df = pd.DataFrame(cat_data, columns=ohe.get_feature_names_out(cst.CATEGORICAL))

    numerical_cols = cst.NUMERICAL
    # Remove signals we won't have access to at prediction time (signals 1 and 3)
    signals_to_remove = {cst.SIGNAL_1, cst.SIGNAL_3}
    numerical_cols = [col for col in numerical_cols if col not in signals_to_remove]

    # Combine with scaled numerical data
    num_df = data[numerical_cols].copy()
    num_df[cst.INCOME] = np.log(
        num_df[cst.INCOME] + 1
    )  # Log-transform income to reduce skewness

    data_preprocessed = pd.concat([cat_df, num_df], axis=1)

    # Return X and y if y exists
    return data_preprocessed, y
