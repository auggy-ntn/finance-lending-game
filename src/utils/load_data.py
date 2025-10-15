# Load data from a CSV file
import pandas as pd

import constants.paths as pth


def load_data(file_path: str = pth.PAST_LOANS_PATH) -> pd.DataFrame:
    """Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file. Defaults to pth.PAST_LOANS_PATH.
    """
    return pd.read_csv(file_path)
