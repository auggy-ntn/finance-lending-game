# Compute the metrics for a given model and add them to a csv file
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import constants.paths as pth


def compute_and_store_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    filepath: Path = pth.METRICS_PATH,
):
    """Compute and store the metrics for a given model.

    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        model_name (str): Name of the model.
        filepath (str, optional): Path to the csv file. Defaults to pth.METRICS_PATH.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame(
        {
            "Model": [model_name],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1-Score": [f1],
        }
    )

    # Append to the csv file or create it if it doesn't exist
    try:
        existing_df = pd.read_csv(filepath)
        updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
        updated_df.to_csv(filepath, index=False)
    except FileNotFoundError:
        metrics_df.to_csv(filepath, index=False)
