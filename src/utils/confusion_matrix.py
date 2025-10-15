import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_test, y_pred, cmap="Greens"):
    """Plot confusion matrix and print classification report.
    Args:
        y_test (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        cmap (str, optional): Colormap for the heatmap. Defaults to "Greens".
    """
    # Evaluation of optimized model
    clf_matrix_optimized = confusion_matrix(y_test, y_pred)

    # Create confusion matrix visualization
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        clf_matrix_optimized,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar_kws={"label": "Count"},
        square=True,
        linewidths=1,
        linecolor="gray",
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(
        "Confusion Matrix - Optimized Model", fontsize=14, fontweight="bold", pad=20
    )

    ax.set_xticklabels(["No Default (0)", "Default (1)"])
    ax.set_yticklabels(["No Default (0)", "Default (1)"], rotation=0)

    tn, fp, fn, tp = clf_matrix_optimized.ravel()
    ax.text(
        0.5,
        -0.15,
        f"True Negatives: {tn}",
        ha="center",
        transform=ax.transAxes,
        fontsize=10,
    )
    ax.text(
        0.5,
        -0.20,
        f"False Positives: {fp}",
        ha="center",
        transform=ax.transAxes,
        fontsize=10,
    )
    ax.text(
        0.5,
        -0.25,
        f"False Negatives: {fn}",
        ha="center",
        transform=ax.transAxes,
        fontsize=10,
    )
    ax.text(
        0.5,
        -0.30,
        f"True Positives: {tp}",
        ha="center",
        transform=ax.transAxes,
        fontsize=10,
    )

    plt.tight_layout()
    plt.show()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print("\nModel Metrics:")
    print(f"{'=' * 40}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")
