from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

EFFICIENCY_CSV = (
    RESULTS_DIR
    / "transformer_freezing_efficiency_comparison.csv"
)

ABLATION_CSV = (
    RESULTS_DIR
    / "transformer_freezing_comparison.csv"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "transformer_freezing_figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK FILES
# ============================================================

if not EFFICIENCY_CSV.exists():
    raise FileNotFoundError(
        f"Missing file:\n{EFFICIENCY_CSV}"
    )

if not ABLATION_CSV.exists():
    raise FileNotFoundError(
        f"Missing file:\n{ABLATION_CSV}"
    )


# ============================================================
# LOAD DATA
# ============================================================

efficiency_df = pd.read_csv(
    EFFICIENCY_CSV
)

ablation_df = pd.read_csv(
    ABLATION_CSV
)


print("=" * 80)
print("TRANSFORMER FREEZING FIGURE GENERATION")
print("=" * 80)

print("\nEfficiency results:")
print(efficiency_df)

print("\nAblation results:")
print(ablation_df)


# ============================================================
# FIGURE 1
# PERFORMANCE VS FROZEN LAYERS
# Uses actual TEST results from ablation study.
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    ablation_df["Frozen_Layers"],
    ablation_df["Accuracy"] * 100,
    marker="o",
    linewidth=2,
    label="Accuracy"
)

plt.plot(
    ablation_df["Frozen_Layers"],
    ablation_df["F1"] * 100,
    marker="o",
    linewidth=2,
    label="F1-score"
)

plt.plot(
    ablation_df["Frozen_Layers"],
    ablation_df["AUC"] * 100,
    marker="o",
    linewidth=2,
    label="ROC-AUC"
)

plt.xlabel(
    "Number of Frozen Encoder Layers"
)

plt.ylabel(
    "Test Performance (%)"
)

plt.title(
    "Effect of Transformer Layer Freezing on Test Performance"
)

plt.xticks(
    ablation_df["Frozen_Layers"]
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

performance_path = (
    FIGURE_DIR
    / "01_freezing_vs_test_performance.png"
)

plt.savefig(
    performance_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# TRAINING TIME VS FROZEN LAYERS
# Uses controlled fixed-epoch benchmark.
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    efficiency_df["frozen_layers"],
    efficiency_df[
        "average_seconds_per_epoch"
    ],
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Number of Frozen Encoder Layers"
)

plt.ylabel(
    "Average Training Time per Epoch (seconds)"
)

plt.title(
    "Training Time Reduction Through Layer Freezing"
)

plt.xticks(
    efficiency_df["frozen_layers"]
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

time_path = (
    FIGURE_DIR
    / "02_freezing_vs_training_time.png"
)

plt.savefig(
    time_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3
# TRAINABLE PARAMETERS VS FROZEN LAYERS
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    efficiency_df["frozen_layers"],
    efficiency_df[
        "trainable_parameters"
    ],
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Number of Frozen Encoder Layers"
)

plt.ylabel(
    "Number of Trainable Parameters"
)

plt.title(
    "Reduction in Trainable Parameters"
)

plt.xticks(
    efficiency_df["frozen_layers"]
)

plt.grid(
    True,
    alpha=0.3
)

plt.ticklabel_format(
    style="plain",
    axis="y"
)

plt.tight_layout()

params_path = (
    FIGURE_DIR
    / "03_freezing_vs_trainable_parameters.png"
)

plt.savefig(
    params_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 4
# GPU MEMORY VS FROZEN LAYERS
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    efficiency_df["frozen_layers"],
    efficiency_df[
        "peak_gpu_allocated_mb"
    ],
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Number of Frozen Encoder Layers"
)

plt.ylabel(
    "Peak GPU Allocated Memory (MB)"
)

plt.title(
    "GPU Memory Usage During Transformer Fine-Tuning"
)

plt.xticks(
    efficiency_df["frozen_layers"]
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

memory_path = (
    FIGURE_DIR
    / "04_freezing_vs_gpu_memory.png"
)

plt.savefig(
    memory_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 5
# PARAMETER REDUCTION VS TIME REDUCTION
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    efficiency_df[
        "trainable_param_reduction_vs_freeze0"
    ],
    efficiency_df[
        "time_reduction_percent"
    ],
    marker="o",
    linewidth=2
)


for _, row in efficiency_df.iterrows():

    plt.annotate(

        f"Freeze {int(row['frozen_layers'])}",

        (
            row[
                "trainable_param_reduction_vs_freeze0"
            ],

            row[
                "time_reduction_percent"
            ]
        ),

        textcoords="offset points",

        xytext=(5, 5)
    )


plt.xlabel(
    "Reduction in Trainable Parameters (%)"
)

plt.ylabel(
    "Training Time Reduction (%)"
)

plt.title(
    "Parameter Reduction vs Training-Time Reduction"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

tradeoff_path = (
    FIGURE_DIR
    / "05_parameter_vs_time_tradeoff.png"
)

plt.savefig(
    tradeoff_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CREATE FINAL COMBINED TABLE
# ============================================================

final_df = pd.DataFrame({

    "Frozen_Layers":
        efficiency_df[
            "frozen_layers"
        ],

    "Trainable_Parameters":
        efficiency_df[
            "trainable_parameters"
        ],

    "Parameter_Reduction_Percent":
        efficiency_df[
            "trainable_param_reduction_vs_freeze0"
        ],

    "Average_Seconds_Per_Epoch":
        efficiency_df[
            "average_seconds_per_epoch"
        ],

    "Time_Reduction_Percent":
        efficiency_df[
            "time_reduction_percent"
        ],

    "Peak_GPU_MB":
        efficiency_df[
            "peak_gpu_allocated_mb"
        ],

    "Validation_Accuracy":
        efficiency_df[
            "validation_accuracy"
        ],

    "Validation_F1":
        efficiency_df[
            "validation_f1"
        ],

    "Validation_AUC":
        efficiency_df[
            "validation_auc"
        ],

    "Test_Accuracy":
        ablation_df[
            "Accuracy"
        ],

    "Test_Precision":
        ablation_df[
            "Precision"
        ],

    "Test_Recall":
        ablation_df[
            "Recall"
        ],

    "Test_F1":
        ablation_df[
            "F1"
        ],

    "Test_AUC":
        ablation_df[
            "AUC"
        ]

})


combined_path = (
    RESULTS_DIR
    / "transformer_freezing_final_comparison.csv"
)

final_df.to_csv(
    combined_path,
    index=False
)


# ============================================================
# FINAL TEXT SUMMARY
# ============================================================

summary_path = (
    RESULTS_DIR
    / "transformer_freezing_final_summary.txt"
)


summary_text = """
============================================================
TRANSFORMER LAYER-FREEZING STUDY — FINAL SUMMARY
============================================================

MODEL:
20-layer Transformer

TOTAL PARAMETERS:
688,130

EXPERIMENTS:
Freeze 0
Freeze 4
Freeze 8
Freeze 12
Freeze 16


------------------------------------------------------------
MAIN PERFORMANCE RESULT
------------------------------------------------------------

Freeze 0:
Test Accuracy = 77.50%
Test F1       = 58.72%
Test AUC      = 79.61%

Freeze 8:
Test Accuracy = 77.00%
Test F1       = 58.18%
Test AUC      = 80.01%

Freeze 12:
Test Accuracy = 76.50%
Test F1       = 58.41%
Test AUC      = 79.55%

Freeze 16:
Test Accuracy = 74.00%
Test F1       = 55.93%
Test AUC      = 79.44%


------------------------------------------------------------
MAIN EFFICIENCY RESULT
------------------------------------------------------------

Freeze 0:
Trainable parameters = 688,130
Average epoch time   = 29.17 sec
Peak GPU memory      = 60.99 MB

Freeze 12:
Trainable parameters = 286,466
Parameter reduction  = 58.37%
Average epoch time   = 25.19 sec
Time reduction       = 13.65%
Peak GPU memory      = 51.97 MB

Freeze 16:
Trainable parameters = 152,578
Parameter reduction  = 77.83%
Average epoch time   = 24.09 sec
Time reduction       = 17.40%
Peak GPU memory      = 48.96 MB


------------------------------------------------------------
INTERPRETATION
------------------------------------------------------------

Increasing the Transformer depth from 2 layers to 20 layers
did not automatically improve generalization.

The 20-layer architecture had much higher model capacity, but
the smaller training dataset made the deeper model more prone
to overfitting.

Progressive layer freezing showed that many early Transformer
encoder layers do not need to be updated during fine-tuning.

Freezing 12 of 20 encoder layers reduced the number of
trainable parameters by approximately 58.4%, while test
accuracy decreased by only 1 percentage point compared with
full fine-tuning.

The controlled efficiency benchmark also showed approximately
13.6% lower training time per epoch for Freeze-12.

Freeze-16 achieved the greatest efficiency:
approximately 77.8% fewer trainable parameters and 17.4%
lower training time per epoch.

However, Freeze-16 produced a larger decline in test
classification performance.

Therefore, Freeze-12 represents a strong balance between
classification performance and computational efficiency.


------------------------------------------------------------
FINAL CONCLUSION
------------------------------------------------------------

The results demonstrate that full fine-tuning of every
Transformer encoder layer is not always necessary for
ECG-based myocardial infarction classification.

A large portion of the early encoder layers can be frozen
while maintaining similar classification performance.

Among the investigated configurations, freezing the first
12 encoder layers provided the strongest overall
performance-efficiency trade-off.
"""


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        summary_text
    )


# ============================================================
# PRINT OUTPUTS
# ============================================================

print("\n" + "=" * 80)
print("FIGURES GENERATED SUCCESSFULLY")
print("=" * 80)

print("\n1.")
print(performance_path)

print("\n2.")
print(time_path)

print("\n3.")
print(params_path)

print("\n4.")
print(memory_path)

print("\n5.")
print(tradeoff_path)

print("\nFinal combined table:")
print(combined_path)

print("\nFinal analysis:")
print(summary_path)

print("\nDone.")