import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_calibration_curve(y_true, y_pred_proba, n_bins=10):
    """Plot the calibration curve for predicted probabilities.

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted probabilities.
        n_bins (int): Number of bins to use for calibration curve.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Calibration curve")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Probability")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid()
    plt.show()
