import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def plot_roc_curve(y_true, y_pred_proba):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true: The true binary labels (0s and 1s).
        y_pred_proba: The predicted probabilities for the positive class (class 1).
    """
    # 1. Calculate the points of the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    # 2. Calculate the Area Under the Curve (AUC)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    # 3. Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.2f})"
    )

    # Plot the random guess line
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess")

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
