from pathlib import Path
import sys

import numpy as np
import pandas as pd
import wfdb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Configuration
# =========================================================

LEAD_NAMES = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6"
]

TRAIN_SAMPLE_SIZE = 800
VAL_SAMPLE_SIZE = 200
TEST_SAMPLE_SIZE = 200

RANDOM_STATE = 42

FIGURE_SIZE = (8, 6)
DPI = 100
Y_LIMIT = 3.0
LINE_WIDTH = 0.8


# =========================================================
# Helper: validate split CSV
# =========================================================

def load_split_csv(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {
        "ecg_id",
        "filename_lr",
        "label",
        "class_name"
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns in {csv_path.name}: "
            f"{sorted(missing)}"
        )

    return df


# =========================================================
# Helper: stratified sample
# =========================================================

def stratified_sample(
    df,
    sample_size,
    random_state=42
):
    sampled_parts = []

    total_rows = len(df)

    for label, group in df.groupby("label"):

        fraction = len(group) / total_rows

        class_sample_size = round(
            sample_size * fraction
        )

        class_sample_size = min(
            class_sample_size,
            len(group)
        )

        sampled_group = group.sample(
            n=class_sample_size,
            random_state=random_state
        )

        sampled_parts.append(
            sampled_group
        )

    sampled_df = pd.concat(
        sampled_parts,
        ignore_index=True
    )

    # Fix rounding if slightly over target
    if len(sampled_df) > sample_size:

        sampled_df = sampled_df.sample(
            n=sample_size,
            random_state=random_state
        )

    return sampled_df.reset_index(
        drop=True
    )


# =========================================================
# ECG grid
# =========================================================

def draw_ecg_grid(
    ax,
    duration,
    y_min,
    y_max
):
    minor_x = np.arange(
        0,
        duration + 1e-6,
        0.04
    )

    major_x = np.arange(
        0,
        duration + 1e-6,
        0.20
    )

    minor_y = np.arange(
        np.floor(y_min / 0.1) * 0.1,
        np.ceil(y_max / 0.1) * 0.1 + 1e-6,
        0.1
    )

    major_y = np.arange(
        np.floor(y_min / 0.5) * 0.5,
        np.ceil(y_max / 0.5) * 0.5 + 1e-6,
        0.5
    )

    for x in minor_x:
        ax.axvline(
            x,
            color="#f2dede",
            linewidth=0.2,
            zorder=0
        )

    for x in major_x:
        ax.axvline(
            x,
            color="#e8bcbc",
            linewidth=0.4,
            zorder=0
        )

    for y in minor_y:
        ax.axhline(
            y,
            color="#f2dede",
            linewidth=0.2,
            zorder=0
        )

    for y in major_y:
        ax.axhline(
            y,
            color="#e8bcbc",
            linewidth=0.4,
            zorder=0
        )


# =========================================================
# Render one ECG
# =========================================================

def render_single_ecg(
    record_path,
    output_path,
    ecg_id
):
    record_path = Path(
        record_path
    )

    output_path = Path(
        output_path
    )

    header_path = Path(
        str(record_path) + ".hea"
    )

    if not header_path.exists():
        raise FileNotFoundError(
            f"Missing header for ECG {ecg_id}: "
            f"{header_path}"
        )

    signal, fields = wfdb.rdsamp(
        str(record_path)
    )

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    if signal.ndim != 2:
        raise ValueError(
            f"Unexpected signal shape "
            f"for ECG {ecg_id}: "
            f"{signal.shape}"
        )

    num_samples, num_leads = signal.shape

    if num_leads != 12:
        raise ValueError(
            f"Expected 12 leads, "
            f"got {num_leads}"
        )

    fs = fields.get(
        "fs",
        100
    )

    duration = (
        num_samples / fs
    )

    time_axis = (
        np.arange(num_samples)
        / fs
    )

    fig, axes = plt.subplots(
        4,
        3,
        figsize=FIGURE_SIZE,
        dpi=DPI
    )

    axes = axes.flatten()

    for lead_idx, ax in enumerate(axes):

        lead_signal = signal[
            :,
            lead_idx
        ]

        draw_ecg_grid(
            ax=ax,
            duration=duration,
            y_min=-Y_LIMIT,
            y_max=Y_LIMIT
        )

        ax.plot(
            time_axis,
            lead_signal,
            color="black",
            linewidth=LINE_WIDTH,
            zorder=2
        )

        ax.set_title(
            LEAD_NAMES[lead_idx],
            fontsize=8,
            fontweight="bold",
            pad=2
        )

        ax.set_xlim(
            0,
            duration
        )

        ax.set_ylim(
            -Y_LIMIT,
            Y_LIMIT
        )

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.96,
        bottom=0.03,
        wspace=0.15,
        hspace=0.25
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.05
    )

    plt.close(fig)


# =========================================================
# Main
# =========================================================

def main():

    PROJECT_ROOT = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    PTBXL_ROOT = (
        PROJECT_ROOT
        / "PTB_XL"
    )

    SPLIT_DIR = (
        PROJECT_ROOT
        / "data"
        / "splits"
    )

    DEMO_SPLIT_DIR = (
        PROJECT_ROOT
        / "data"
        / "demo_splits"
    )

    OUTPUT_DIR = (
        PROJECT_ROOT
        / "data"
        / "rendered_images"
    )

    RESULTS_DIR = (
        PROJECT_ROOT
        / "results"
    )

    DEMO_SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # Load full PTB-XL split CSVs
    # -----------------------------------------------------

    train_df = load_split_csv(
        SPLIT_DIR / "train.csv"
    )

    val_df = load_split_csv(
        SPLIT_DIR / "val.csv"
    )

    test_df = load_split_csv(
        SPLIT_DIR / "test.csv"
    )


    # -----------------------------------------------------
    # Create demo subsets
    # -----------------------------------------------------

    train_sample = stratified_sample(
        train_df,
        TRAIN_SAMPLE_SIZE,
        RANDOM_STATE
    )

    val_sample = stratified_sample(
        val_df,
        VAL_SAMPLE_SIZE,
        RANDOM_STATE
    )

    test_sample = stratified_sample(
        test_df,
        TEST_SAMPLE_SIZE,
        RANDOM_STATE
    )


    # -----------------------------------------------------
    # Save demo split CSVs
    # -----------------------------------------------------

    train_sample.to_csv(
        DEMO_SPLIT_DIR / "train.csv",
        index=False
    )

    val_sample.to_csv(
        DEMO_SPLIT_DIR / "val.csv",
        index=False
    )

    test_sample.to_csv(
        DEMO_SPLIT_DIR / "test.csv",
        index=False
    )


    # -----------------------------------------------------
    # Print distributions
    # -----------------------------------------------------

    print("=" * 60)
    print("DEMO DATASET DISTRIBUTION")
    print("=" * 60)

    print("\nTrain:")
    print(
        train_sample[
            "class_name"
        ].value_counts()
    )

    print("\nValidation:")
    print(
        val_sample[
            "class_name"
        ].value_counts()
    )

    print("\nTest:")
    print(
        test_sample[
            "class_name"
        ].value_counts()
    )


    # -----------------------------------------------------
    # Combine demo records
    # -----------------------------------------------------

    records_df = pd.concat(
        [
            train_sample,
            val_sample,
            test_sample
        ],
        ignore_index=True
    )

    records_df = (
        records_df
        .drop_duplicates(
            subset="ecg_id"
        )
        .reset_index(
            drop=True
        )
    )

    total_records = len(
        records_df
    )

    print(
        "\nTotal demo ECG records:",
        total_records
    )


    # -----------------------------------------------------
    # Render ECG images
    # -----------------------------------------------------

    rendered_count = 0
    skipped_count = 0
    failed_records = []

    for i, row in enumerate(
        records_df.itertuples(
            index=False
        ),
        start=1
    ):

        ecg_id = int(
            row.ecg_id
        )

        record_path = (
            PTBXL_ROOT
            / row.filename_lr
        )

        output_path = (
            OUTPUT_DIR
            / f"{ecg_id}.png"
        )

        # Reuse any images generated
        # during your previous full run
        if output_path.exists():

            skipped_count += 1

        else:

            try:

                render_single_ecg(
                    record_path,
                    output_path,
                    ecg_id
                )

                rendered_count += 1

            except Exception as e:

                failed_records.append({
                    "ecg_id": ecg_id,
                    "filename_lr":
                        row.filename_lr,
                    "error": str(e)
                })


        if (
            i % 50 == 0
            or i == total_records
        ):

            print(
                f"Processed "
                f"{i}/{total_records} | "
                f"Rendered: "
                f"{rendered_count} | "
                f"Skipped: "
                f"{skipped_count} | "
                f"Failed: "
                f"{len(failed_records)}"
            )


    # -----------------------------------------------------
    # Save failures
    # -----------------------------------------------------

    if failed_records:

        failure_df = pd.DataFrame(
            failed_records
        )

        failure_df.to_csv(
            RESULTS_DIR
            / "demo_render_failures.csv",
            index=False
        )


    # -----------------------------------------------------
    # Final verification
    # -----------------------------------------------------

    missing_images = []

    for row in records_df.itertuples(
        index=False
    ):

        image_path = (
            OUTPUT_DIR
            / f"{int(row.ecg_id)}.png"
        )

        if not image_path.exists():

            missing_images.append(
                int(row.ecg_id)
            )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("DEMO RENDER SUMMARY")
    print("=" * 60)

    print(
        "Demo ECG records:",
        total_records
    )

    print(
        "Newly rendered:",
        rendered_count
    )

    print(
        "Reused existing:",
        skipped_count
    )

    print(
        "Failures:",
        len(failed_records)
    )

    print(
        "Missing final images:",
        len(missing_images)
    )

    print(
        "\nDemo split CSVs saved to:"
    )

    print(
        DEMO_SPLIT_DIR
    )

    print(
        "\nRendered images:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\nRendering stopped by user."
        )

        sys.exit(1)