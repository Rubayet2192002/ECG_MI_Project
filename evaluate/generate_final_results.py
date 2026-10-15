from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Expected files
# =========================================================

CNN_RESULTS = (
    RESULTS_DIR
    / "cnn_demo_results.json"
)

TRANSFORMER_RESULTS = (
    RESULTS_DIR
    / "transformer_demo_results.json"
)

HYBRID_RESULTS = (
    RESULTS_DIR
    / "hybrid_demo_results.json"
)


CNN_PREDICTIONS = (
    RESULTS_DIR
    / "cnn_demo_test_predictions.csv"
)

TRANSFORMER_PREDICTIONS = (
    RESULTS_DIR
    / "transformer_demo_test_predictions.csv"
)

HYBRID_PREDICTIONS = (
    RESULTS_DIR
    / "hybrid_demo_test_predictions.csv"
)


CNN_HISTORY = (
    RESULTS_DIR
    / "cnn_demo_history.csv"
)

TRANSFORMER_HISTORY = (
    RESULTS_DIR
    / "transformer_demo_history.csv"
)

HYBRID_HISTORY = (
    RESULTS_DIR
    / "hybrid_demo_history.csv"
)


# =========================================================
# Utility
# =========================================================

def check_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def load_json(path):

    check_file(path)

    with open(
        path,
        "r"
    ) as f:

        return json.load(f)


# =========================================================
# Load model results
# =========================================================

cnn = load_json(
    CNN_RESULTS
)

transformer = load_json(
    TRANSFORMER_RESULTS
)

hybrid = load_json(
    HYBRID_RESULTS
)


# =========================================================
# Build final comparison table
# =========================================================

comparison = pd.DataFrame({

    "Model": [
        "CNN (ResNet18)",
        "Transformer",
        "Hybrid"
    ],

    "Accuracy": [
        cnn["test_accuracy"],
        transformer["test_accuracy"],
        hybrid["test_accuracy"]
    ],

    "Precision": [
        cnn["test_precision"],
        transformer["test_precision"],
        hybrid["test_precision"]
    ],

    "Recall": [
        cnn["test_recall"],
        transformer["test_recall"],
        hybrid["test_recall"]
    ],

    "F1_Score": [
        cnn["test_f1"],
        transformer["test_f1"],
        hybrid["test_f1"]
    ],

    "ROC_AUC": [
        cnn["test_auc"],
        transformer["test_auc"],
        hybrid["test_auc"]
    ]
})


# Percent version for presentation
comparison_percent = comparison.copy()

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]:

    comparison_percent[column] = (
        comparison_percent[column]
        * 100
    ).round(2)


comparison_path = (
    RESULTS_DIR
    / "final_model_comparison.csv"
)

comparison_percent.to_csv(
    comparison_path,
    index=False
)


print("=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

print(
    comparison_percent
    .to_string(
        index=False
    )
)


# =========================================================
# Figure 1:
# Performance comparison bar chart
# =========================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

models = (
    comparison["Model"]
    .tolist()
)

x = np.arange(
    len(metrics)
)

width = 0.25


fig, ax = plt.subplots(
    figsize=(11, 6)
)


for i, model in enumerate(models):

    values = (
        comparison[
            comparison["Model"]
            == model
        ][metrics]
        .iloc[0]
        .values
        * 100
    )

    ax.bar(
        x + (
            i - 1
        ) * width,
        values,
        width,
        label=model
    )


ax.set_ylabel(
    "Score (%)"
)

ax.set_xlabel(
    "Evaluation Metric"
)

ax.set_title(
    "CNN vs Transformer vs Hybrid Performance"
)

ax.set_xticks(
    x
)

ax.set_xticklabels([
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
])

ax.set_ylim(
    0,
    100
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()


performance_chart_path = (
    FIGURE_DIR
    / "model_performance_comparison.png"
)

plt.savefig(
    performance_chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# Load predictions
# =========================================================

check_file(
    CNN_PREDICTIONS
)

check_file(
    TRANSFORMER_PREDICTIONS
)

check_file(
    HYBRID_PREDICTIONS
)


cnn_pred = pd.read_csv(
    CNN_PREDICTIONS
)

transformer_pred = pd.read_csv(
    TRANSFORMER_PREDICTIONS
)

hybrid_pred = pd.read_csv(
    HYBRID_PREDICTIONS
)


# =========================================================
# Verify same labels
# =========================================================

cnn_labels = (
    cnn_pred[
        "true_label"
    ].to_numpy()
)

transformer_labels = (
    transformer_pred[
        "true_label"
    ].to_numpy()
)

hybrid_labels = (
    hybrid_pred[
        "true_label"
    ].to_numpy()
)


if not np.array_equal(
    cnn_labels,
    transformer_labels
):

    raise ValueError(
        "CNN and Transformer test labels "
        "do not match."
    )


if not np.array_equal(
    cnn_labels,
    hybrid_labels
):

    raise ValueError(
        "CNN and Hybrid test labels "
        "do not match."
    )


# =========================================================
# Figure 2:
# ROC curves
# =========================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)


prediction_sets = [

    (
        "CNN",
        cnn_pred
    ),

    (
        "Transformer",
        transformer_pred
    ),

    (
        "Hybrid",
        hybrid_pred
    )
]


for model_name, prediction_df in prediction_sets:

    labels = (
        prediction_df[
            "true_label"
        ]
        .to_numpy()
    )

    probabilities = (
        prediction_df[
            "mi_probability"
        ]
        .to_numpy()
    )

    fpr, tpr, _ = roc_curve(
        labels,
        probabilities
    )

    model_auc = auc(
        fpr,
        tpr
    )

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"{model_name} "
            f"(AUC = {model_auc:.3f})"
        )
    )


ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random classifier"
)


ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.set_title(
    "ROC Curves for MI Classification"
)

ax.legend(
    loc="lower right"
)

ax.grid(
    alpha=0.25
)

plt.tight_layout()


roc_path = (
    FIGURE_DIR
    / "roc_curve_comparison.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# Confusion matrix function
# =========================================================

def save_confusion_matrix(
    prediction_df,
    model_name,
    filename
):

    labels = (
        prediction_df[
            "true_label"
        ].to_numpy()
    )

    predictions = (
        prediction_df[
            "predicted_label"
        ].to_numpy()
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    display = (
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Non-MI",
                "MI"
            ]
        )
    )

    display.plot(
        ax=ax,
        values_format="d",
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(
        f"{model_name} Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# Figures 3–5:
# Confusion matrices
# =========================================================

save_confusion_matrix(
    cnn_pred,
    "CNN (ResNet18)",
    "cnn_confusion_matrix.png"
)

save_confusion_matrix(
    transformer_pred,
    "Transformer",
    "transformer_confusion_matrix.png"
)

save_confusion_matrix(
    hybrid_pred,
    "Hybrid CNN + Transformer",
    "hybrid_confusion_matrix.png"
)


# =========================================================
# Training curve function
# =========================================================

def save_training_curves(
    history_path,
    model_name,
    filename
):

    check_file(
        history_path
    )

    history = pd.read_csv(
        history_path
    )

    required_columns = {
        "epoch",
        "train_loss",
        "val_loss",
        "train_f1",
        "val_f1",
        "train_auc",
        "val_auc"
    }

    missing = (
        required_columns
        - set(history.columns)
    )

    if missing:

        raise ValueError(
            f"{model_name} history is missing: "
            f"{sorted(missing)}"
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5)
    )


    # ---------------------------------------------
    # Loss
    # ---------------------------------------------

    axes[0].plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        label="Train"
    )

    axes[0].plot(
        history["epoch"],
        history["val_loss"],
        marker="o",
        label="Validation"
    )

    axes[0].set_title(
        "Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.25
    )


    # ---------------------------------------------
    # F1
    # ---------------------------------------------

    axes[1].plot(
        history["epoch"],
        history["train_f1"],
        marker="o",
        label="Train"
    )

    axes[1].plot(
        history["epoch"],
        history["val_f1"],
        marker="o",
        label="Validation"
    )

    axes[1].set_title(
        "F1 Score"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "F1"
    )

    axes[1].set_ylim(
        0,
        1
    )

    axes[1].legend()

    axes[1].grid(
        alpha=0.25
    )


    # ---------------------------------------------
    # AUC
    # ---------------------------------------------

    axes[2].plot(
        history["epoch"],
        history["train_auc"],
        marker="o",
        label="Train"
    )

    axes[2].plot(
        history["epoch"],
        history["val_auc"],
        marker="o",
        label="Validation"
    )

    axes[2].set_title(
        "ROC-AUC"
    )

    axes[2].set_xlabel(
        "Epoch"
    )

    axes[2].set_ylabel(
        "AUC"
    )

    axes[2].set_ylim(
        0,
        1
    )

    axes[2].legend()

    axes[2].grid(
        alpha=0.25
    )


    fig.suptitle(
        f"{model_name} Training History",
        fontsize=14
    )

    plt.tight_layout()


    output_path = (
        FIGURE_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# Figures 6–8:
# Training curves
# =========================================================

save_training_curves(
    CNN_HISTORY,
    "CNN (ResNet18)",
    "cnn_training_curves.png"
)

save_training_curves(
    TRANSFORMER_HISTORY,
    "Transformer",
    "transformer_training_curves.png"
)

save_training_curves(
    HYBRID_HISTORY,
    "Hybrid CNN + Transformer",
    "hybrid_training_curves.png"
)


# =========================================================
# Generate concise text summary
# =========================================================

summary_path = (
    RESULTS_DIR
    / "final_results_summary.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PTB-XL MI CLASSIFICATION - "
        "PRELIMINARY RESULTS\n"
    )

    f.write(
        "=" * 60
    )

    f.write("\n\n")

    f.write(
        "Dataset subset:\n"
    )

    f.write(
        "Train: 800 ECGs\n"
    )

    f.write(
        "Validation: 200 ECGs\n"
    )

    f.write(
        "Test: 200 ECGs\n\n"
    )

    f.write(
        comparison_percent
        .to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "Observations:\n"
    )

    f.write(
        "- Hybrid produced the highest "
        "overall accuracy.\n"
    )

    f.write(
        "- Hybrid produced the highest "
        "MI precision.\n"
    )

    f.write(
        "- Transformer produced the highest "
        "MI recall/sensitivity.\n"
    )

    f.write(
        "- Transformer produced the highest "
        "MI F1 score.\n"
    )

    f.write(
        "- Hybrid produced the highest "
        "ROC-AUC.\n"
    )

    f.write(
        "- Results are preliminary and were "
        "obtained from a stratified PTB-XL "
        "subset rather than the entire dataset.\n"
    )


# =========================================================
# Completion
# =========================================================

print("\n" + "=" * 75)

print(
    "FINAL RESULTS GENERATED"
)

print("=" * 75)

print(
    "\nComparison table:"
)

print(
    comparison_path
)

print(
    "\nFigures directory:"
)

print(
    FIGURE_DIR
)

print(
    "\nSummary:"
)

print(
    summary_path
)


print(
    "\nGenerated figures:"
)

for path in sorted(
    FIGURE_DIR.glob("*.png")
):

    print(
        " -",
        path.name
    )