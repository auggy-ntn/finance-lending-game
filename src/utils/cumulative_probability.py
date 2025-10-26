import matplotlib.pyplot as plt
import numpy as np


def plot_cumulative_probability(y_pred_proba):
    """
    Plots the cumulative count of individuals against their sorted predicted probability
    of default.

    Args:
        y_pred_proba: An array of predicted probabilities for the positive class.
    """
    # 1. Sort the probabilities in ascending order
    sorted_probas = np.sort(y_pred_proba)

    # 2. Create the cumulative count for the y-axis
    cumulative_count = np.arange(1, len(sorted_probas) + 1)

    # 3. Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_probas, cumulative_count, linewidth=2)

    # Add labels and title
    plt.xlabel("Predicted Probability of Default")
    plt.ylabel("Cumulative Number of Individuals")
    plt.title("Cumulative Distribution of Predicted Probabilities")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Set limits to make the plot clean
    plt.xlim([0.0, 1.0])
    plt.ylim([0, len(sorted_probas)])

    plt.show()
