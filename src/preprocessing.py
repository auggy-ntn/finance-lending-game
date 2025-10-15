# Preprocessing for model training
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import constants.constants as cst


def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess the data for model training.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The preprocessed data.
        pd.Series: The target variable.
    """

    # One-hot encode categorical variables
    ohe = OneHotEncoder(sparse_output=False, drop="first")
    cat_data = ohe.fit_transform(data[cst.CATEGORICAL])
    cat_df = pd.DataFrame(cat_data, columns=ohe.get_feature_names_out(cst.CATEGORICAL))

    # Combine with scaled numerical data
    scaler = StandardScaler()
    num_data = scaler.fit_transform(data[cst.NUMERICAL])
    num_df = pd.DataFrame(num_data, columns=cst.NUMERICAL)

    data_preprocessed = pd.concat([cat_df, num_df], axis=1)

    # Drop signals we won't have access to at prediction time (signals 1 and 3)
    data_preprocessed = data_preprocessed.drop(columns=[cst.SIGNAL_1, cst.SIGNAL_3])

    # Return X and y
    return data_preprocessed, data[cst.TARGET]
