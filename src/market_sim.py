from itertools import product
from typing import Dict, Tuple

import pandas as pd

import constants.constants as cst
from src.strategies import predict_rates


def _get_winner(df: pd.DataFrame) -> pd.DataFrame:
    """Determines the winner for a loan, i.e. the bank the client accepts the loan from.

    Args:
        df (pd.DataFrame): The dataframe containing loan interest rates from the 3 banks

    Returns:
        pd.DataFrame: A dataframe with an additional column 'winner' indicating
            the winning bank.
    """
    # Apply borrower type discount (2% discount for preferred bank)
    adjusted_rate_1 = df[cst.BANK_1_RATE].where(
        df[cst.BORROWER_TYPE] != 1, df[cst.BANK_1_RATE] - 0.02
    )
    adjusted_rate_2 = df[cst.BANK_2_RATE].where(
        df[cst.BORROWER_TYPE] != 2, df[cst.BANK_2_RATE] - 0.02
    )
    adjusted_rate_3 = df[cst.BANK_3_RATE].where(
        df[cst.BORROWER_TYPE] != 3, df[cst.BANK_3_RATE] - 0.02
    )

    # Create dataframe with adjusted rates
    rates_df = pd.DataFrame(
        {
            cst.BANK_1_RATE: adjusted_rate_1,
            cst.BANK_2_RATE: adjusted_rate_2,
            cst.BANK_3_RATE: adjusted_rate_3,
        }
    )

    # Find minimum rate
    min_rate = rates_df.min(axis=1)

    # Count how many banks have the minimum rate (for tie detection)
    ties = (rates_df == min_rate.values[:, None]).sum(axis=1)

    # Get the bank with minimum rate (idxmin returns column name)
    df[cst.WINNER] = rates_df.idxmin(axis=1)

    # Mark ties as "no_winner"
    df.loc[ties > 1, cst.WINNER] = "no_winner"

    return df


def compute_profit(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the profits of each loan for each bank.

    Args:
        df (pd.DataFrame): The dataframe containing loan data, and the
            default realizations.

    Returns:
        pd.DataFrame: A dataframe containing the profits for each bank.
    """
    df = df.copy()
    df = _get_winner(df)
    banks = [cst.BANK_1_RATE, cst.BANK_2_RATE, cst.BANK_3_RATE]

    for bank in banks:
        profit_col = f"profit_{bank}"

        # Vectorized profit calculation
        # Start with all NaN (not applicable)
        df[profit_col] = pd.NA

        # For winners only, calculate profit
        winner_mask = df[cst.WINNER] == bank

        # When bank wins and no default: profit = rate * loan_amount
        no_default_mask = winner_mask & (df[cst.TARGET] == 0)
        df.loc[no_default_mask, profit_col] = (
            df.loc[no_default_mask, bank] * cst.LOAN_AMOUNT
        )

        # When bank wins and default occurs: loss = -loan_amount
        default_mask = winner_mask & (df[cst.TARGET] == 1)
        df.loc[default_mask, profit_col] = -cst.LOAN_AMOUNT

    return df


def optimize_rate_params(
    predictions_df: pd.DataFrame,
    market_results_df: pd.DataFrame,
    linear_param_grid: Dict[str, list],
    quadratic_param_grid: Dict[str, list],
    strategy: str = "both",
    verbose: bool = True,
) -> Tuple[Dict, Dict, pd.DataFrame]:
    """Optimize linear and quadratic rate parameters through market simulation.

    This function tests different parameter combinations for linear and/or quadratic
    rate strategies by simulating market outcomes and computing profits. It returns
    the best parameters for each strategy.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing loan predictions with
            default probabilities. Must include ID and DEFAULT_PROBABILITY columns.
        market_results_df (pd.DataFrame): DataFrame containing market simulation data
            including competitor rates, borrower types, and loan outcomes. Must include
            BANK_1_RATE, BANK_3_RATE, BORROWER_TYPE, and TARGET columns.
        linear_param_grid (Dict[str, list]): Parameter grid for linear strategy.
            Expected keys: 'alpha_0', 'alpha_1', 'delta'.
            Example: {'alpha_0': [0.1, 0.2], 'alpha_1': [0.05, 0.1], 'delta': [0.05]}
        quadratic_param_grid (Dict[str, list]): Parameter grid for quadratic strategy.
            Expected keys: 'beta_0', 'beta_1', 'beta_2', 'gamma'.
            Example: {'beta_0': [0.1, 0.2], 'beta_1': [0.05, 0.1],
                     'beta_2': [0.1, 0.2], 'gamma': [0.05]}
        strategy (str): Which strategy to optimize.
            Options: 'linear', 'quadratic', 'both'.
            Default is 'both'.
        verbose (bool): If True, prints progress during optimization.
            Default is True.

    Returns:
        Tuple containing:
            - best_linear_params (Dict): Best parameters for linear strategy
            - best_quadratic_params (Dict): Best parameters for quadratic
                strategy
            - results_df (pd.DataFrame): DataFrame with all tested
                combinations and profits

    Example:
        >>> linear_grid = {
        ...     'alpha_0': [0.1, 0.15, 0.2],
        ...     'alpha_1': [0.05, 0.1, 0.15],
        ...     'delta': [0.05]
        ... }
        >>> quadratic_grid = {
        ...     'beta_0': [0.1, 0.15, 0.2],
        ...     'beta_1': [0.05, 0.1],
        ...     'beta_2': [0.1, 0.2],
        ...     'gamma': [0.05]
        ... }
        >>> best_lin, best_quad, results = optimize_rate_params(
        ...     predictions, market_data, linear_grid, quadratic_grid
        ... )
    """
    results = []

    # Optimize linear strategy
    if strategy in ["linear", "both"]:
        if verbose:
            print("Optimizing linear strategy...")

        # Generate all combinations of linear parameters
        param_names = list(linear_param_grid.keys())
        param_values = [linear_param_grid[name] for name in param_names]

        total_combinations = len(list(product(*param_values)))

        for i, param_combo in enumerate(product(*param_values)):
            linear_params = dict(zip(param_names, param_combo, strict=True))

            # Predict rates with current parameters
            temp_predictions = predict_rates(
                predictions_df.copy(),
                linear_params=linear_params,
                quadratic_params={"beta_0": 0, "beta_1": 0, "beta_2": 0, "gamma": 0},
            )

            # Merge with market results
            comparison = market_results_df.copy()
            comparison[cst.BANK_2_RATE] = temp_predictions[cst.LINEAR_RATE].values

            # Compute profit
            comparison = compute_profit(comparison)
            total_profit = comparison[f"profit_{cst.BANK_2_RATE}"].sum()

            # Store results
            result = {
                "strategy": "linear",
                **linear_params,
                "total_profit": total_profit,
            }
            results.append(result)

            if verbose and (i + 1) % max(1, total_combinations // 10) == 0:
                print(f"  Tested {i + 1}/{total_combinations} linear combinations")

    # Optimize quadratic strategy
    if strategy in ["quadratic", "both"]:
        if verbose:
            print("Optimizing quadratic strategy...")

        # Generate all combinations of quadratic parameters
        param_names = list(quadratic_param_grid.keys())
        param_values = [quadratic_param_grid[name] for name in param_names]

        total_combinations = len(list(product(*param_values)))

        for i, param_combo in enumerate(product(*param_values)):
            quadratic_params = dict(zip(param_names, param_combo, strict=True))

            # Predict rates with current parameters
            temp_predictions = predict_rates(
                predictions_df.copy(),
                linear_params={"alpha_0": 0, "alpha_1": 0, "delta": 0},
                quadratic_params=quadratic_params,
            )

            # Merge with market results
            comparison = market_results_df.copy()
            comparison[cst.BANK_2_RATE] = temp_predictions[cst.QUADRATIC_RATE].values

            # Compute profit
            comparison = compute_profit(comparison)
            total_profit = comparison[f"profit_{cst.BANK_2_RATE}"].sum()

            # Store results
            result = {
                "strategy": "quadratic",
                **quadratic_params,
                "total_profit": total_profit,
            }
            results.append(result)

            if verbose and (i + 1) % max(1, total_combinations // 10) == 0:
                print(f"  Tested {i + 1}/{total_combinations} quadratic combinations")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Find best parameters for each strategy
    best_linear_params = {}
    best_quadratic_params = {}

    if strategy in ["linear", "both"]:
        linear_results = results_df[results_df["strategy"] == "linear"]
        best_linear_idx = linear_results["total_profit"].idxmax()
        best_linear_row = linear_results.loc[best_linear_idx]
        best_linear_params = {
            "alpha_0": best_linear_row["alpha_0"],
            "alpha_1": best_linear_row["alpha_1"],
            "delta": best_linear_row["delta"],
        }
        if verbose:
            print(f"\nBest linear parameters: {best_linear_params}")
            print(f"Best linear profit: ${best_linear_row['total_profit']:,.2f}")

    if strategy in ["quadratic", "both"]:
        quadratic_results = results_df[results_df["strategy"] == "quadratic"]
        best_quadratic_idx = quadratic_results["total_profit"].idxmax()
        best_quadratic_row = quadratic_results.loc[best_quadratic_idx]
        best_quadratic_params = {
            "beta_0": best_quadratic_row["beta_0"],
            "beta_1": best_quadratic_row["beta_1"],
            "beta_2": best_quadratic_row["beta_2"],
            "gamma": best_quadratic_row["gamma"],
        }
        if verbose:
            print(f"\nBest quadratic parameters: {best_quadratic_params}")
            print(f"Best quadratic profit: ${best_quadratic_row['total_profit']:,.2f}")

    return best_linear_params, best_quadratic_params, results_df


def analyze_adverse_selection(
    predictions_df: pd.DataFrame,
    market_results_df: pd.DataFrame,
    our_rates: pd.Series,
    bank_name: str = cst.BANK_2_RATE,
) -> pd.DataFrame:
    """Analyze adverse selection patterns in historical lending data.

    This function helps identify if you're experiencing winner's curse by
    analyzing the relationship between winning loans and their characteristics.

    Args:
        predictions_df (pd.DataFrame): DataFrame with default probability
            predictions.
        market_results_df (pd.DataFrame): Historical market results with
            competitor rates and outcomes.
        our_rates (pd.Series): The interest rates you offered.
        bank_name (str): Your bank identifier. Default is BANK_2_RATE.

    Returns:
        pd.DataFrame: Analysis results showing selection bias metrics.
    """
    df = market_results_df.copy()
    df["our_rate"] = our_rates.values
    df["default_probability"] = predictions_df[cst.DEFAULT_PROBABILITY].values

    # Calculate competitor minimum rate
    competitor_cols = [
        col
        for col in [cst.BANK_1_RATE, cst.BANK_2_RATE, cst.BANK_3_RATE]
        if col != bank_name
    ]
    df["min_competitor_rate"] = df[competitor_cols].min(axis=1)

    # Calculate how much cheaper/expensive we are
    df["rate_gap"] = df["our_rate"] - df["min_competitor_rate"]

    # Add winner information
    df = _get_winner(df)
    df["we_won"] = df[cst.WINNER] == bank_name

    # Analyze by rate gap quintiles
    df["rate_gap_quintile"] = pd.qcut(
        df["rate_gap"],
        q=5,
        labels=["Much Cheaper", "Cheaper", "Similar", "More Exp", "Much More Exp"],
        duplicates="drop",
    )

    # Group analysis
    analysis = (
        df.groupby("rate_gap_quintile", observed=True)
        .agg(
            {
                "we_won": ["mean", "count"],  # Win rate and count
                # Default rate when we won
                cst.TARGET: lambda x: x[df.loc[x.index, "we_won"]].mean(),
                # Predicted risk when we won
                "default_probability": lambda x: x[df.loc[x.index, "we_won"]].mean(),
            }
        )
        .round(4)
    )

    analysis.columns = [
        "win_rate",
        "loan_count",
        "actual_default_rate_when_won",
        "predicted_default_rate_when_won",
    ]

    return analysis


def analyze_risk_segment_performance(
    predictions_df: pd.DataFrame,
    market_results_df: pd.DataFrame,
    our_rates: pd.Series,
    bank_name: str = cst.BANK_2_RATE,
    n_segments: int = 5,
) -> pd.DataFrame:
    """Analyze performance across different risk segments.

    Identifies if certain risk segments are more profitable or if you're
    experiencing adverse selection in specific risk buckets.

    Args:
        predictions_df (pd.DataFrame): DataFrame with default probability
            predictions.
        market_results_df (pd.DataFrame): Historical market results.
        our_rates (pd.Series): The interest rates you offered.
        bank_name (str): Your bank identifier. Default is BANK_2_RATE.
        n_segments (int): Number of risk segments to create. Default is 5.

    Returns:
        pd.DataFrame: Performance metrics by risk segment.
    """
    df = market_results_df.copy()
    df["our_rate"] = our_rates.values
    df["default_probability"] = predictions_df[cst.DEFAULT_PROBABILITY].values

    # Add winner information
    df = _get_winner(df)
    df["we_won"] = df[cst.WINNER] == bank_name

    # Create risk segments
    df["risk_segment"] = pd.qcut(
        df["default_probability"],
        q=n_segments,
        labels=[f"Q{i + 1}" for i in range(n_segments)],
        duplicates="drop",
    )

    # Calculate profit for our won loans
    df["our_profit"] = 0.0
    won_mask = df["we_won"]
    df.loc[won_mask & (df[cst.TARGET] == 0), "our_profit"] = (
        df.loc[won_mask & (df[cst.TARGET] == 0), "our_rate"] * cst.LOAN_AMOUNT
    )
    df.loc[won_mask & (df[cst.TARGET] == 1), "our_profit"] = -cst.LOAN_AMOUNT

    # Group analysis
    analysis = (
        df.groupby("risk_segment", observed=True)
        .agg(
            {
                "default_probability": ["mean", "min", "max"],
                "we_won": ["sum", "mean"],
                cst.TARGET: lambda x: x[df.loc[x.index, "we_won"]].mean(),
                "our_profit": ["sum", "mean"],
                "our_rate": lambda x: x[df.loc[x.index, "we_won"]].mean(),
            }
        )
        .round(4)
    )

    analysis.columns = [
        "avg_pred_default",
        "min_pred_default",
        "max_pred_default",
        "loans_won",
        "win_rate",
        "actual_default_rate",
        "total_profit",
        "avg_profit_per_loan",
        "avg_rate_charged",
    ]

    # Calculate expected vs actual performance
    analysis["expected_default_rate"] = analysis["avg_pred_default"]
    analysis["prediction_error"] = (
        analysis["actual_default_rate"] - analysis["expected_default_rate"]
    )

    return analysis


def suggest_risk_based_markup(
    adverse_selection_analysis: pd.DataFrame,
    risk_segment_analysis: pd.DataFrame,
    base_markup: float = 0.05,
) -> Dict[str, float]:
    """Suggest markup adjustments based on adverse selection analysis.

    Args:
        adverse_selection_analysis (pd.DataFrame): Output from
            analyze_adverse_selection.
        risk_segment_analysis (pd.DataFrame): Output from
            analyze_risk_segment_performance.
        base_markup (float): Base markup to adjust from.

    Returns:
        Dict: Suggested markup parameters with reasoning.
    """
    suggestions = {"base_markup": base_markup, "adjustments": {}, "reasoning": []}

    # Check for adverse selection in cheap pricing
    if len(adverse_selection_analysis) > 0:
        much_cheaper = adverse_selection_analysis.iloc[0]
        actual_default = much_cheaper["actual_default_rate_when_won"]
        predicted_default = much_cheaper["predicted_default_rate_when_won"]
        if actual_default > predicted_default * 1.1:
            suggestions["adjustments"]["adverse_selection_penalty"] = 0.02
            suggestions["reasoning"].append(
                "⚠️  Adverse selection detected: When pricing much cheaper, "
                "actual defaults exceed predictions by >10%. "
                "Consider adding 2% markup."
            )

    # Check for systematic under/overperformance by risk segment
    if len(risk_segment_analysis) > 0:
        for idx, row in risk_segment_analysis.iterrows():
            if row["prediction_error"] > 0.05:  # Underestimating risk
                suggestions["reasoning"].append(
                    f"⚠️  Risk segment {idx}: Underestimating defaults by "
                    f"{row['prediction_error']:.2%}. Consider higher markup."
                )
            elif row["avg_profit_per_loan"] < 0:
                suggestions["reasoning"].append(
                    f"❌ Risk segment {idx}: Negative profit. "
                    f"Consider avoiding or increasing markup significantly."
                )

    return suggestions
