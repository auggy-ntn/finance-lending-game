# Preprocessing for model training
from typing import Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import constants.constants as cst
import constants.paths as pth
from src.utils.load_data import load_data


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


class Preprocessor:
    """A singleton class to preprocess data for model training and prediction."""

    def __init__(self):
        self.ohe = OneHotEncoder(sparse_output=False, drop="first")
        self.scaler = StandardScaler()
        self.fitted = False

    def fit_transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit the preprocessor and transform the data.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The preprocessed data.
            pd.Series: The target variable.
        """
        # One-hot encode categorical variables
        cat_data = self.ohe.fit_transform(data[cst.CATEGORICAL])
        cat_df = pd.DataFrame(
            cat_data, columns=self.ohe.get_feature_names_out(cst.CATEGORICAL)
        )

        # Combine with scaled numerical data
        numerical_cols_drop_signals = cst.NUMERICAL.copy()
        numerical_cols_drop_signals.remove(cst.SIGNAL_1)
        numerical_cols_drop_signals.remove(cst.SIGNAL_3)

        num_data = self.scaler.fit_transform(data[numerical_cols_drop_signals])
        num_df = pd.DataFrame(num_data, columns=numerical_cols_drop_signals)

        data_preprocessed = pd.concat([cat_df, num_df], axis=1)

        self.fitted = True

        # Return X and y
        return data_preprocessed, data[cst.TARGET]

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using the fitted preprocessor.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        cat_data = self.ohe.transform(data[cst.CATEGORICAL])
        cat_df = pd.DataFrame(
            cat_data, columns=self.ohe.get_feature_names_out(cst.CATEGORICAL)
        )

        numerical_cols_drop_signals = cst.NUMERICAL.copy()
        numerical_cols_drop_signals.remove(cst.SIGNAL_1)
        numerical_cols_drop_signals.remove(cst.SIGNAL_3)

        num_data = self.scaler.transform(data[numerical_cols_drop_signals])
        num_df = pd.DataFrame(num_data, columns=numerical_cols_drop_signals)

        data_preprocessed = pd.concat([cat_df, num_df], axis=1)

        return data_preprocessed


if __name__ == "__main__":
    # Train the preprocessor on training data and save it
    train_data = load_data()
    train_data = train_data.drop(columns=[cst.SIGNAL_1, cst.SIGNAL_3])
    preprocessor = Preprocessor()
    X_train, y_train = preprocessor.fit_transform(train_data)
    joblib.dump(preprocessor, pth.PREPROCESSOR_PATH)
