import numpy as np
import pandas as pd

import constants.constants as cst


def _break_even_rate(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate the break-even interest rate given default probabilities.

    The break-even interest rate is the rate at which the expected return
    on a loan is zero, considering the probability of default.

    Args:
        predictions (pd.DataFrame): DataFrame containing default probabilities
            for loans. The DataFrame has an id column and a default
            probability column.

    Returns:
        pd.DataFrame: DataFrame with an additional column for the
            break-even interest rate.
    """
    predictions = predictions.copy()
    predictions[cst.BREAK_EVEN_RATE] = predictions[cst.DEFAULT_PROBABILITY] / (
        1 - predictions[cst.DEFAULT_PROBABILITY]
    )
    return predictions


def _linear_strategy(
    predictions: pd.DataFrame, alpha_0: float, alpha_1: float, delta: float = 0.05
) -> pd.DataFrame:
    """Calculate final interest rates using a linear strategy.

    The linear strategy adds a fixed base rate to the break-even rate
    to determine the final interest rate for each loan.
    The formula used is:
        linear_rate = break_even_rate + alpha_0 + (alpha_1 * default_probability)
    Additionally, a minimum markup (delta) over the break-even rate is enforced.
        final_rate = max(linear_rate, (1 + delta) * break_even_rate)


    Args:
        predictions (pd.DataFrame): DataFrame containing break-even rates.
        alpha_0 (float): The intercept (base rate) of the linear model.
        alpha_1 (float): The slope (coefficient) of the linear model.
        delta (float): Minimum markup percentage over the break-even rate.

    Returns:
        pd.DataFrame: DataFrame with an additional column for the
            final interest rate.
    """
    predictions = predictions.copy()
    predictions[cst.LINEAR_RATE] = (
        predictions[cst.BREAK_EVEN_RATE]
        + alpha_0
        + (alpha_1 * predictions[cst.DEFAULT_PROBABILITY])
    )
    predictions[cst.LINEAR_RATE] = np.maximum(
        predictions[cst.LINEAR_RATE], (1 + delta) * predictions[cst.BREAK_EVEN_RATE]
    )

    predictions[cst.LINEAR_RATE] = predictions[cst.LINEAR_RATE].clip(upper=1.0)
    return predictions


def _quadratic_strategy(
    predictions: pd.DataFrame,
    beta_0: float,
    beta_1: float,
    beta_2: float,
    gamma: float = 0.05,
) -> pd.DataFrame:
    """Calculate final interest rates using a quadratic strategy.

    The quadratic strategy uses a quadratic function of the default
    probability to determine the final interest rate for each loan.
    The formula used is:
        quadratic_rate = break_even_rate + beta_0 + (beta_1 * default_probability)
                           + (beta_2 * default_probability^2)
    Additionally, a minimum markup (gamma) over the break-even rate is enforced.
        final_rate = max(quadratic_rate, (1 + gamma) * break_even_rate)

    Args:
        predictions (pd.DataFrame): DataFrame containing break-even rates.
        beta_0 (float): The intercept (base rate) of the quadratic model.
        beta_1 (float): The linear coefficient of the quadratic model.
        beta_2 (float): The quadratic coefficient of the quadratic model.
        gamma (float): Minimum markup percentage over the break-even rate.

    Returns:
        pd.DataFrame: DataFrame with an additional column for the
            final interest rate.
    """
    predictions = predictions.copy()
    predictions[cst.QUADRATIC_RATE] = (
        predictions[cst.BREAK_EVEN_RATE]
        + beta_0
        + (beta_1 * predictions[cst.DEFAULT_PROBABILITY])
        + (beta_2 * predictions[cst.DEFAULT_PROBABILITY] ** 2)
    )
    predictions[cst.QUADRATIC_RATE] = np.maximum(
        predictions[cst.QUADRATIC_RATE], (1 + gamma) * predictions[cst.BREAK_EVEN_RATE]
    )
    predictions[cst.QUADRATIC_RATE] = predictions[cst.QUADRATIC_RATE].clip(upper=1.0)
    return predictions


def predict_rates(
    predictions: pd.DataFrame,
    linear_params: dict,
    quadratic_params: dict,
) -> pd.DataFrame:
    """Predict break-even rates and final interest rates using
    linear and quadratic strategies.

    Args:
        predictions (pd.DataFrame): DataFrame containing default probabilities.
        linear_params (dict): Parameters for the linear strategy.
        quadratic_params (dict): Parameters for the quadratic strategy.

    Returns:
        pd.DataFrame: DataFrame with additional columns for break-even rates,
            linear rates, and quadratic rates.
    """
    predictions = _break_even_rate(predictions)
    predictions = _linear_strategy(predictions, **linear_params)
    predictions = _quadratic_strategy(predictions, **quadratic_params)
    return predictions


def risk_tiered_strategy(
    predictions: pd.DataFrame,
    low_risk_markup: float,
    medium_risk_markup: float,
    high_risk_markup: float,
    low_threshold: float = 0.2,
    high_threshold: float = 0.4,
    min_delta: float = 0.05,
) -> pd.DataFrame:
    """Apply risk-tiered markup strategy to mitigate winner's curse.

    This strategy applies different markups based on predicted risk levels,
    being more conservative with high-risk loans where model uncertainty
    and adverse selection are more likely.

    Args:
        predictions (pd.DataFrame): DataFrame with default probabilities
            and break-even rates.
        low_risk_markup (float): Markup for low-risk loans (can be aggressive).
        medium_risk_markup (float): Markup for medium-risk loans.
        high_risk_markup (float): Markup for high-risk loans (conservative).
        low_threshold (float): Default probability threshold for low risk.
            Default is 0.2.
        high_threshold (float): Default probability threshold for high risk.
            Default is 0.4.
        min_delta (float): Minimum markup percentage. Default is 0.05.

    Returns:
        pd.DataFrame: DataFrame with risk-tiered rate column.
    """
    predictions = predictions.copy()

    # Ensure break-even rate is calculated
    if cst.BREAK_EVEN_RATE not in predictions.columns:
        predictions = _break_even_rate(predictions)

    # Initialize the tiered rate
    predictions["risk_tiered_rate"] = predictions[cst.BREAK_EVEN_RATE]

    # Apply markups based on risk tiers
    low_risk_mask = predictions[cst.DEFAULT_PROBABILITY] < low_threshold
    high_risk_mask = predictions[cst.DEFAULT_PROBABILITY] >= high_threshold
    medium_risk_mask = ~low_risk_mask & ~high_risk_mask

    predictions.loc[low_risk_mask, "risk_tiered_rate"] = (
        predictions.loc[low_risk_mask, cst.BREAK_EVEN_RATE] + low_risk_markup
    )
    predictions.loc[medium_risk_mask, "risk_tiered_rate"] = (
        predictions.loc[medium_risk_mask, cst.BREAK_EVEN_RATE] + medium_risk_markup
    )
    predictions.loc[high_risk_mask, "risk_tiered_rate"] = (
        predictions.loc[high_risk_mask, cst.BREAK_EVEN_RATE] + high_risk_markup
    )

    # Enforce minimum markup
    predictions["risk_tiered_rate"] = np.maximum(
        predictions["risk_tiered_rate"],
        (1 + min_delta) * predictions[cst.BREAK_EVEN_RATE],
    )

    # Cap at 100%
    predictions["risk_tiered_rate"] = predictions["risk_tiered_rate"].clip(upper=1.0)

    return predictions
