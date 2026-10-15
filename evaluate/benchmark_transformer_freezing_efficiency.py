from pathlib import Path
import sys
import random
import time
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ptbxl_dataset import PTBXLDataset
from src.signal_preprocessing import ECGSignalPreprocessor
from models.transformer import ECGTransformer


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 8

NUM_WORKERS = 0

# IMPORTANT:
# Every experiment runs EXACTLY 5 epochs.
FIXED_EPOCHS = 5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

GRADIENT_CLIP = 1.0


FREEZE_SETTINGS = [
    0,
    4,
    8,
    12,
    16,
]


# ============================================================
# TRANSFORMER CONFIGURATION
# ============================================================

NUM_LEADS = 12

SIGNAL_LENGTH = 1000

PATCH_SIZE = 10

D_MODEL = 64

NUM_HEADS = 4

NUM_LAYERS = 20

DIM_FEEDFORWARD = 128

DROPOUT = 0.1

NUM_CLASSES = 2


# ============================================================
# PATHS
# ============================================================

PTBXL_ROOT = (
    PROJECT_ROOT
    / "PTB_XL"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "demo_splits"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

BENCHMARK_DIR = (
    RESULTS_DIR
    / "transformer_freezing_efficiency"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "transformer_20layer_full_best.pt"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(csv_path):

    df = pd.read_csv(csv_path)

    counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    if 0 not in counts.index or 1 not in counts.index:

        raise ValueError(
            "Expected binary labels 0 and 1."
        )


    total = len(df)

    weight_0 = (
        total
        / (
            2 * counts.loc[0]
        )
    )

    weight_1 = (
        total
        / (
            2 * counts.loc[1]
        )
    )


    return torch.tensor(

        [
            weight_0,
            weight_1
        ],

        dtype=torch.float32
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions,
    probabilities
):

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )


    try:

        auc = roc_auc_score(
            labels,
            probabilities
        )

    except ValueError:

        auc = float("nan")


    return {

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "auc":
            auc
    }


# ============================================================
# FIND TRANSFORMER ENCODER
# ============================================================

def find_transformer_encoder(model):

    for name, module in model.named_modules():

        if isinstance(
            module,
            nn.TransformerEncoder
        ):

            return name, module


    raise RuntimeError(
        "Could not find nn.TransformerEncoder "
        "inside ECGTransformer."
    )


# ============================================================
# FREEZE EARLY ENCODER BLOCKS
# ============================================================

def freeze_encoder_layers(
    model,
    num_layers_to_freeze
):

    encoder_name, encoder = (
        find_transformer_encoder(model)
    )


    total_layers = len(
        encoder.layers
    )


    if (
        num_layers_to_freeze
        > total_layers
    ):

        raise ValueError(
            f"Cannot freeze "
            f"{num_layers_to_freeze} layers. "
            f"Model has only "
            f"{total_layers} layers."
        )


    # First make everything trainable.

    for parameter in model.parameters():

        parameter.requires_grad = True


    # Freeze first N encoder layers.

    for layer_index in range(
        num_layers_to_freeze
    ):

        layer = (
            encoder.layers[
                layer_index
            ]
        )

        for parameter in (
            layer.parameters()
        ):

            parameter.requires_grad = False


    return (
        encoder_name,
        total_layers
    )


# ============================================================
# PARAMETER COUNTS
# ============================================================

def count_parameters(model):

    total = sum(

        parameter.numel()

        for parameter
        in model.parameters()

    )


    trainable = sum(

        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad

    )


    frozen = (
        total
        - trainable
    )


    return (
        total,
        trainable,
        frozen
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(device):

    model = ECGTransformer(

        num_leads=NUM_LEADS,

        signal_length=SIGNAL_LENGTH,

        patch_size=PATCH_SIZE,

        d_model=D_MODEL,

        num_heads=NUM_HEADS,

        num_layers=NUM_LAYERS,

        dim_feedforward=DIM_FEEDFORWARD,

        dropout=DROPOUT,

        num_classes=NUM_CLASSES

    )


    return model.to(device)


# ============================================================
# LOAD PRETRAINED 20-LAYER CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    device
):

    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=device,

        weights_only=False

    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    return checkpoint


# ============================================================
# CREATE TRAIN LOADER
# ============================================================

def create_train_loader(
    train_dataset,
    device
):

    # New generator for every experiment.
    # This ensures identical shuffle order.

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED
    )


    return DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=(
            device.type == "cuda"
        ),

        generator=generator

    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp
):

    model.train()


    running_loss = 0.0

    labels_all = []

    predictions_all = []

    probabilities_all = []


    for signals, labels in loader:

        signals = signals.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.autocast(

            device_type=device.type,

            dtype=torch.float16,

            enabled=use_amp

        ):

            logits = model(
                signals
            )

            loss = criterion(
                logits,
                labels
            )


        scaler.scale(
            loss
        ).backward()


        scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(

            [
                parameter

                for parameter
                in model.parameters()

                if parameter.requires_grad
            ],

            GRADIENT_CLIP

        )


        scaler.step(
            optimizer
        )

        scaler.update()


        running_loss += (

            loss.item()
            * signals.size(0)

        )


        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]


        predictions = torch.argmax(
            logits,
            dim=1
        )


        labels_all.extend(

            labels
            .detach()
            .cpu()
            .numpy()
            .tolist()

        )


        predictions_all.extend(

            predictions
            .detach()
            .cpu()
            .numpy()
            .tolist()

        )


        probabilities_all.extend(

            probabilities
            .detach()
            .cpu()
            .numpy()
            .tolist()

        )


    metrics = calculate_metrics(

        labels_all,

        predictions_all,

        probabilities_all

    )


    metrics["loss"] = (

        running_loss
        / len(loader.dataset)

    )


    return metrics


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp
):

    model.eval()


    running_loss = 0.0

    labels_all = []

    predictions_all = []

    probabilities_all = []


    for signals, labels in loader:

        signals = signals.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        with torch.autocast(

            device_type=device.type,

            dtype=torch.float16,

            enabled=use_amp

        ):

            logits = model(
                signals
            )

            loss = criterion(
                logits,
                labels
            )


        running_loss += (

            loss.item()
            * signals.size(0)

        )


        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]


        predictions = torch.argmax(
            logits,
            dim=1
        )


        labels_all.extend(

            labels
            .detach()
            .cpu()
            .numpy()
            .tolist()

        )


        predictions_all.extend(

            predictions
            .detach()
            .cpu()
            .numpy()
            .tolist()

        )


        probabilities_all.extend(

            probabilities
            .detach()
            .cpu()
            .numpy()
            .tolist()

        )


    metrics = calculate_metrics(

        labels_all,

        predictions_all,

        probabilities_all

    )


    metrics["loss"] = (

        running_loss
        / len(loader.dataset)

    )


    return metrics


# ============================================================
# RUN ONE BENCHMARK
# ============================================================

def benchmark_configuration(
    frozen_layers,
    train_dataset,
    val_loader,
    class_weights,
    device,
    use_amp
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"EFFICIENCY BENCHMARK — "
        f"FREEZE {frozen_layers}"
    )

    print(
        "=" * 90
    )


    # Reset everything before each run.

    set_seed(
        RANDOM_SEED
    )


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        device
    )


    load_checkpoint(
        model,
        device
    )


    (
        encoder_name,
        total_layers

    ) = freeze_encoder_layers(

        model,

        frozen_layers

    )


    (
        total_params,
        trainable_params,
        frozen_params

    ) = count_parameters(
        model
    )


    parameter_reduction_percent = (

        (
            total_params
            - trainable_params
        )

        / total_params

        * 100

    )


    print(
        "Encoder:",
        encoder_name
    )

    print(
        "Total encoder layers:",
        total_layers
    )

    print(
        "Frozen encoder layers:",
        frozen_layers
    )

    print(
        "Trainable encoder layers:",
        total_layers - frozen_layers
    )


    print(
        "\nTotal parameters:",
        f"{total_params:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_params:,}"
    )

    print(
        "Frozen parameters:",
        f"{frozen_params:,}"
    )

    print(
        "Parameter reduction:",
        f"{parameter_reduction_percent:.2f}%"
    )


    # --------------------------------------------------------
    # Fresh train loader
    # --------------------------------------------------------

    train_loader = create_train_loader(

        train_dataset,

        device

    )


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(

        weight=class_weights

    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(

        [

            parameter

            for parameter
            in model.parameters()

            if parameter.requires_grad

        ],

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=use_amp

    )


    # --------------------------------------------------------
    # GPU memory reset
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats(
            device
        )


    # --------------------------------------------------------
    # Warm-up
    #
    # One forward/backward batch is done BEFORE timing.
    #
    # This reduces timing distortion from CUDA initialization.
    # --------------------------------------------------------

    print(
        "\nRunning untimed warm-up batch..."
    )


    model.train()


    warmup_signals, warmup_labels = (
        next(iter(train_loader))
    )


    warmup_signals = (
        warmup_signals.to(
            device
        )
    )

    warmup_labels = (
        warmup_labels.to(
            device
        )
    )


    optimizer.zero_grad(
        set_to_none=True
    )


    with torch.autocast(

        device_type=device.type,

        dtype=torch.float16,

        enabled=use_amp

    ):

        warmup_logits = model(
            warmup_signals
        )

        warmup_loss = criterion(

            warmup_logits,

            warmup_labels

        )


    scaler.scale(
        warmup_loss
    ).backward()


    scaler.step(
        optimizer
    )

    scaler.update()


    optimizer.zero_grad(
        set_to_none=True
    )


    del warmup_signals
    del warmup_labels
    del warmup_logits
    del warmup_loss


    if torch.cuda.is_available():

        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(
            device
        )


    # Recreate loader so warm-up does not consume
    # one shuffled batch from timed training.

    train_loader = create_train_loader(

        train_dataset,

        device

    )


    # ========================================================
    # EXACTLY 5 TIMED TRAINING EPOCHS
    # ========================================================

    epoch_times = []

    history = []


    print(
        f"\nStarting fixed "
        f"{FIXED_EPOCHS}-epoch benchmark..."
    )


    for epoch in range(
        1,
        FIXED_EPOCHS + 1
    ):

        if torch.cuda.is_available():

            torch.cuda.synchronize()


        epoch_start = (
            time.perf_counter()
        )


        train_metrics = train_one_epoch(

            model=model,

            loader=train_loader,

            criterion=criterion,

            optimizer=optimizer,

            scaler=scaler,

            device=device,

            use_amp=use_amp

        )


        if torch.cuda.is_available():

            torch.cuda.synchronize()


        epoch_end = (
            time.perf_counter()
        )


        epoch_seconds = (
            epoch_end
            - epoch_start
        )


        epoch_times.append(
            epoch_seconds
        )


        history.append({

            "epoch":
                epoch,

            "seconds":
                epoch_seconds,

            "loss":
                train_metrics[
                    "loss"
                ],

            "accuracy":
                train_metrics[
                    "accuracy"
                ],

            "precision":
                train_metrics[
                    "precision"
                ],

            "recall":
                train_metrics[
                    "recall"
                ],

            "f1":
                train_metrics[
                    "f1"
                ],

            "auc":
                train_metrics[
                    "auc"
                ]

        })


        print(

            f"Epoch "
            f"{epoch:02d}/{FIXED_EPOCHS} | "

            f"Time: "
            f"{epoch_seconds:.2f} sec | "

            f"Loss: "
            f"{train_metrics['loss']:.4f} | "

            f"Acc: "
            f"{train_metrics['accuracy']:.4f} | "

            f"F1: "
            f"{train_metrics['f1']:.4f} | "

            f"AUC: "
            f"{train_metrics['auc']:.4f}"

        )


    # ========================================================
    # TIMING RESULTS
    # ========================================================

    total_training_seconds = sum(
        epoch_times
    )


    average_epoch_seconds = (
        total_training_seconds
        / FIXED_EPOCHS
    )


    min_epoch_seconds = min(
        epoch_times
    )


    max_epoch_seconds = max(
        epoch_times
    )


    # ========================================================
    # GPU MEMORY
    # ========================================================

    if torch.cuda.is_available():

        peak_allocated_mb = (

            torch.cuda
            .max_memory_allocated(
                device
            )

            / 1024**2

        )


        peak_reserved_mb = (

            torch.cuda
            .max_memory_reserved(
                device
            )

            / 1024**2

        )

    else:

        peak_allocated_mb = 0.0

        peak_reserved_mb = 0.0


    # ========================================================
    # FINAL VALIDATION
    #
    # IMPORTANT:
    # Validation is NOT included in training timing.
    # ========================================================

    print(
        "\nRunning untimed validation..."
    )


    val_metrics = evaluate(

        model=model,

        loader=val_loader,

        criterion=criterion,

        device=device,

        use_amp=use_amp

    )


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "frozen_layers":
            frozen_layers,

        "trainable_layers":
            total_layers
            - frozen_layers,

        "total_parameters":
            total_params,

        "trainable_parameters":
            trainable_params,

        "frozen_parameters":
            frozen_params,

        "parameter_reduction_percent":
            parameter_reduction_percent,

        "fixed_epochs":
            FIXED_EPOCHS,

        "total_training_seconds":
            total_training_seconds,

        "total_training_minutes":
            total_training_seconds
            / 60,

        "average_seconds_per_epoch":
            average_epoch_seconds,

        "minimum_epoch_seconds":
            min_epoch_seconds,

        "maximum_epoch_seconds":
            max_epoch_seconds,

        "peak_gpu_allocated_mb":
            peak_allocated_mb,

        "peak_gpu_reserved_mb":
            peak_reserved_mb,

        "validation_accuracy":
            val_metrics[
                "accuracy"
            ],

        "validation_precision":
            val_metrics[
                "precision"
            ],

        "validation_recall":
            val_metrics[
                "recall"
            ],

        "validation_f1":
            val_metrics[
                "f1"
            ],

        "validation_auc":
            val_metrics[
                "auc"
            ]

    }


    # ========================================================
    # SAVE INDIVIDUAL HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history
    )


    history_path = (

        BENCHMARK_DIR
        / (
            f"freeze_{frozen_layers}"
            "_timing_history.csv"
        )

    )


    history_df.to_csv(

        history_path,

        index=False

    )


    # ========================================================
    # SAVE INDIVIDUAL RESULT
    # ========================================================

    result_path = (

        BENCHMARK_DIR
        / (
            f"freeze_{frozen_layers}"
            "_efficiency.json"
        )

    )


    with open(
        result_path,
        "w"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        "\n"
        + "-" * 90
    )

    print(
        f"FREEZE {frozen_layers} SUMMARY"
    )

    print(
        "-" * 90
    )


    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )


    print(
        f"Parameter reduction: "
        f"{parameter_reduction_percent:.2f}%"
    )


    print(
        f"Average epoch time: "
        f"{average_epoch_seconds:.2f} sec"
    )


    print(
        f"Total 5-epoch training time: "
        f"{total_training_seconds:.2f} sec"
    )


    print(
        f"Peak GPU allocated memory: "
        f"{peak_allocated_mb:.2f} MB"
    )


    print(
        f"Peak GPU reserved memory: "
        f"{peak_reserved_mb:.2f} MB"
    )


    print(
        f"Validation Accuracy: "
        f"{val_metrics['accuracy']:.4f}"
    )


    print(
        f"Validation F1: "
        f"{val_metrics['f1']:.4f}"
    )


    print(
        f"Validation AUC: "
        f"{val_metrics['auc']:.4f}"
    )


    # Free memory.

    del model
    del optimizer
    del scaler


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return result


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )


    BENCHMARK_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    # ========================================================
    # CHECK REQUIRED FILES
    # ========================================================

    train_csv = (
        SPLIT_DIR
        / "train.csv"
    )

    val_csv = (
        SPLIT_DIR
        / "val.csv"
    )


    for path in [

        train_csv,

        val_csv,

        CHECKPOINT_PATH

    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    use_amp = (
        device.type == "cuda"
    )


    print(
        "=" * 90
    )

    print(
        "TRANSFORMER FREEZING — "
        "FIXED 5-EPOCH EFFICIENCY BENCHMARK"
    )

    print(
        "=" * 90
    )


    print(
        "Device:",
        device
    )


    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )


    print(
        "\nFreeze settings:",
        FREEZE_SETTINGS
    )


    print(
        "Fixed epochs:",
        FIXED_EPOCHS
    )


    print(
        "Batch size:",
        BATCH_SIZE
    )


    print(
        "Learning rate:",
        LEARNING_RATE
    )


    # ========================================================
    # PREPROCESSING
    # ========================================================

    signal_transform = (
        ECGSignalPreprocessor(

            normalize=True,

            clip_value=5.0

        )
    )


    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = PTBXLDataset(

        csv_file=train_csv,

        ptbxl_root=PTBXL_ROOT,

        transform=signal_transform

    )


    val_dataset = PTBXLDataset(

        csv_file=val_csv,

        ptbxl_root=PTBXL_ROOT,

        transform=signal_transform

    )


    print(
        "\nTrain samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )


    # ========================================================
    # VALIDATION LOADER
    # ========================================================

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            device.type == "cuda"
        )

    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = (

        calculate_class_weights(
            train_csv
        )

        .to(device)

    )


    print(
        "\nClass weights:",
        class_weights
        .detach()
        .cpu()
        .tolist()
    )


    # ========================================================
    # RUN BENCHMARKS
    # ========================================================

    results = []


    benchmark_start = (
        time.time()
    )


    for frozen_layers in FREEZE_SETTINGS:

        result = benchmark_configuration(

            frozen_layers=
                frozen_layers,

            train_dataset=
                train_dataset,

            val_loader=
                val_loader,

            class_weights=
                class_weights,

            device=
                device,

            use_amp=
                use_amp

        )


        results.append(
            result
        )


    benchmark_runtime = (

        time.time()
        - benchmark_start

    )


    # ========================================================
    # CREATE COMPARISON DATAFRAME
    # ========================================================

    comparison_df = pd.DataFrame(
        results
    )


    # ========================================================
    # CALCULATE SPEEDUP RELATIVE TO FREEZE 0
    # ========================================================

    freeze_zero_row = (

        comparison_df[

            comparison_df[
                "frozen_layers"
            ] == 0

        ]

        .iloc[0]

    )


    baseline_time = (
        freeze_zero_row[
            "average_seconds_per_epoch"
        ]
    )


    baseline_trainable_params = (
        freeze_zero_row[
            "trainable_parameters"
        ]
    )


    comparison_df[
        "time_reduction_percent"
    ] = (

        (
            baseline_time
            - comparison_df[
                "average_seconds_per_epoch"
            ]
        )

        / baseline_time

        * 100

    )


    comparison_df[
        "speedup_factor"
    ] = (

        baseline_time

        / comparison_df[
            "average_seconds_per_epoch"
        ]

    )


    comparison_df[
        "trainable_param_reduction_vs_freeze0"
    ] = (

        (
            baseline_trainable_params

            - comparison_df[
                "trainable_parameters"
            ]
        )

        / baseline_trainable_params

        * 100

    )


    # ========================================================
    # SAVE FINAL CSV
    # ========================================================

    comparison_path = (

        RESULTS_DIR
        / "transformer_freezing_efficiency_comparison.csv"

    )


    comparison_df.to_csv(

        comparison_path,

        index=False

    )


    # ========================================================
    # PRINT FINAL TABLE
    # ========================================================

    print(
        "\n\n"
        + "=" * 115
    )

    print(
        "FINAL FIXED-EPOCH EFFICIENCY COMPARISON"
    )

    print(
        "=" * 115
    )


    display_columns = [

        "frozen_layers",

        "trainable_parameters",

        "trainable_param_reduction_vs_freeze0",

        "average_seconds_per_epoch",

        "time_reduction_percent",

        "speedup_factor",

        "peak_gpu_allocated_mb",

        "validation_accuracy",

        "validation_f1",

        "validation_auc",

    ]


    print(

        comparison_df[
            display_columns
        ].to_string(

            index=False,

            float_format=lambda x: (
                f"{x:.4f}"
            )

        )

    )


    print(
        "\nTotal benchmark runtime:"
    )

    print(
        f"{benchmark_runtime / 60:.2f} minutes"
    )


    print(
        "\nFinal CSV:"
    )

    print(
        comparison_path
    )


    print(
        "\nIndividual benchmark results:"
    )

    print(
        BENCHMARK_DIR
    )


if __name__ == "__main__":

    main()