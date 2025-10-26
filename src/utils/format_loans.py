# Format csv for final submission

import pandas as pd

import constants.constants as cst
import constants.paths as pth


def format_loans_for_submission(
    loans: pd.DataFrame, strategy: str, save: bool = True, round: int = 1
) -> pd.DataFrame:
    """Format loans DataFrame for submission.

    Args:
        loans (pd.DataFrame): DataFrame containing the id of each loan and their rates
                                for different strategies.
        strategy (str): The strategy used for rate prediction. Options are
                        'linear', 'quadratic'.
        save (bool): Whether to save the formatted DataFrame to a CSV file.
                        Default is True.
        round (int): The round number for naming the output file. Default is 1.

    Returns:
        pd.DataFrame: Formatted DataFrame ready for submission.
    """
    formatted_loans = loans.copy()
    if strategy == "linear":
        rate_col = cst.LINEAR_RATE
    elif strategy == "quadratic":
        rate_col = cst.QUADRATIC_RATE
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    formatted_loans = formatted_loans[[cst.ID, rate_col]]
    formatted_loans = formatted_loans.rename(
        columns={
            rate_col: "rate",
        }
    )
    if save:
        formatted_loans.to_csv(
            pth.OUTPUT_DIR / f"submission_round_{round}.csv", index=False
        )
    return formatted_loans
