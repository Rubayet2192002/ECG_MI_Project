from pathlib import Path
import sys
import random
import json
import time

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
    confusion_matrix,
    classification_report,
)


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.ptbxl_dataset import PTBXLDataset
from src.signal_preprocessing import ECGSignalPreprocessor
from models.transformer import ECGTransformer


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_SEED = 42

# Same demo subset used previously
BATCH_SIZE = 8

NUM_WORKERS = 0

EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 5

GRADIENT_CLIP = 1.0


# =========================================================
# TRANSFORMER CONFIGURATION
# =========================================================

NUM_LAYERS = 20

NUM_LEADS = 12

SIGNAL_LENGTH = 1000

PATCH_SIZE = 10

D_MODEL = 64

NUM_HEADS = 4

DIM_FEEDFORWARD = 128

DROPOUT = 0.1

NUM_CLASSES = 2


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# =========================================================
# CLASS WEIGHTS
# =========================================================

def calculate_class_weights(
    csv_path
):

    df = pd.read_csv(
        csv_path
    )

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
            2
            * counts.loc[0]
        )
    )

    weight_1 = (
        total
        / (
            2
            * counts.loc[1]
        )
    )

    return torch.tensor(
        [
            weight_0,
            weight_1
        ],
        dtype=torch.float32
    )


# =========================================================
# METRICS
# =========================================================

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


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp
):

    model.train()

    running_loss = 0.0

    all_labels = []

    all_predictions = []

    all_probabilities = []


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
            model.parameters(),
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


        all_labels.extend(

            labels.detach()
            .cpu()
            .numpy()
            .tolist()

        )


        all_predictions.extend(

            predictions.detach()
            .cpu()
            .numpy()
            .tolist()

        )


        all_probabilities.extend(

            probabilities.detach()
            .cpu()
            .numpy()
            .tolist()

        )


    epoch_loss = (

        running_loss
        / len(loader.dataset)

    )


    metrics = calculate_metrics(

        all_labels,

        all_predictions,

        all_probabilities

    )


    metrics["loss"] = (
        epoch_loss
    )


    return metrics


# =========================================================
# EVALUATION
# =========================================================

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


    all_labels = []

    all_predictions = []

    all_probabilities = []


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


        all_labels.extend(

            labels.detach()
            .cpu()
            .numpy()
            .tolist()

        )


        all_predictions.extend(

            predictions.detach()
            .cpu()
            .numpy()
            .tolist()

        )


        all_probabilities.extend(

            probabilities.detach()
            .cpu()
            .numpy()
            .tolist()

        )


    epoch_loss = (

        running_loss
        / len(loader.dataset)

    )


    metrics = calculate_metrics(

        all_labels,

        all_predictions,

        all_probabilities

    )


    metrics["loss"] = (
        epoch_loss
    )


    return (

        metrics,

        all_labels,

        all_predictions,

        all_probabilities

    )


# =========================================================
# GPU MEMORY TEST
# =========================================================

def run_memory_test(
    model,
    device
):

    print(
        "\nRunning 20-layer forward-pass test..."
    )


    model.eval()


    dummy_signal = torch.randn(

        BATCH_SIZE,

        NUM_LEADS,

        SIGNAL_LENGTH,

        device=device

    )


    try:

        with torch.no_grad():

            with torch.autocast(

                device_type=device.type,

                dtype=torch.float16,

                enabled=(
                    device.type == "cuda"
                )

            ):

                output = model(
                    dummy_signal
                )


        print(
            "Memory test passed."
        )

        print(
            "Dummy input:",
            dummy_signal.shape
        )

        print(
            "Output:",
            output.shape
        )


        if device.type == "cuda":

            allocated = (

                torch.cuda
                .memory_allocated()
                / 1024**2

            )


            reserved = (

                torch.cuda
                .memory_reserved()
                / 1024**2

            )


            print(
                f"GPU allocated memory: "
                f"{allocated:.2f} MB"
            )

            print(
                f"GPU reserved memory: "
                f"{reserved:.2f} MB"
            )


        del dummy_signal
        del output


        if device.type == "cuda":

            torch.cuda.empty_cache()


    except torch.OutOfMemoryError:

        print(
            "\nCUDA OUT OF MEMORY "
            "during the forward test."
        )

        print(
            "Reduce BATCH_SIZE from 8 to 4."
        )

        raise


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed(
        RANDOM_SEED
    )


    # =====================================================
    # PATHS
    # =====================================================

    PTBXL_ROOT = (

        PROJECT_ROOT
        / "PTB_XL"

    )


    SPLIT_DIR = (

        PROJECT_ROOT
        / "data"
        / "demo_splits"

    )


    CHECKPOINT_DIR = (

        PROJECT_ROOT
        / "checkpoints"

    )


    RESULTS_DIR = (

        PROJECT_ROOT
        / "results"

    )


    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    train_csv = (
        SPLIT_DIR
        / "train.csv"
    )


    val_csv = (
        SPLIT_DIR
        / "val.csv"
    )


    test_csv = (
        SPLIT_DIR
        / "test.csv"
    )


    for path in [

        train_csv,

        val_csv,

        test_csv

    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing split file: {path}"
            )


    # =====================================================
    # DEVICE
    # =====================================================

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"

    )


    use_amp = (
        device.type == "cuda"
    )


    print(
        "=" * 75
    )

    print(
        "20-LAYER TRANSFORMER BASELINE"
    )

    print(
        "=" * 75
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


        gpu_memory = (

            torch.cuda
            .get_device_properties(0)
            .total_memory

            / 1024**3

        )


        print(
            f"GPU memory: "
            f"{gpu_memory:.2f} GB"
        )


    print(
        "Mixed precision:",
        use_amp
    )


    print(
        "\nTransformer configuration"
    )


    print(
        "Encoder layers:",
        NUM_LAYERS
    )


    print(
        "Embedding dimension:",
        D_MODEL
    )


    print(
        "Attention heads:",
        NUM_HEADS
    )


    print(
        "Feed-forward dimension:",
        DIM_FEEDFORWARD
    )


    print(
        "Patch size:",
        PATCH_SIZE
    )


    print(
        "Batch size:",
        BATCH_SIZE
    )


    # =====================================================
    # PREPROCESSING
    # =====================================================

    signal_transform = (
        ECGSignalPreprocessor(

            normalize=True,

            clip_value=5.0

        )
    )


    # =====================================================
    # DATASETS
    # =====================================================

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


    test_dataset = PTBXLDataset(

        csv_file=test_csv,

        ptbxl_root=PTBXL_ROOT,

        transform=signal_transform

    )


    print(
        "\nDataset sizes"
    )


    print(
        "Train:",
        len(train_dataset)
    )


    print(
        "Validation:",
        len(val_dataset)
    )


    print(
        "Test:",
        len(test_dataset)
    )


    # =====================================================
    # DATALOADERS
    # =====================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=(
            device.type == "cuda"
        )

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            device.type == "cuda"
        )

    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            device.type == "cuda"
        )

    )


    # =====================================================
    # MODEL
    # =====================================================

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


    model = model.to(
        device
    )


    total_params = sum(

        p.numel()

        for p in model.parameters()

    )


    trainable_params = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

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
        f"{total_params - trainable_params:,}"
    )


    # =====================================================
    # MEMORY TEST
    # =====================================================

    run_memory_test(
        model,
        device
    )


    # =====================================================
    # CLASS WEIGHTS
    # =====================================================

    class_weights = (

        calculate_class_weights(
            train_csv
        )

        .to(
            device
        )

    )


    print(
        "\nClass weights:",
        class_weights
        .detach()
        .cpu()
        .tolist()
    )


    # =====================================================
    # LOSS
    # =====================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    # =====================================================
    # SCHEDULER
    # =====================================================

    scheduler = (

        torch.optim.lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=2

        )

    )


    # =====================================================
    # AMP
    # =====================================================

    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=use_amp

    )


    # =====================================================
    # TRAINING
    # =====================================================

    best_val_f1 = -1.0

    epochs_without_improvement = 0

    history = []


    best_checkpoint_path = (

        CHECKPOINT_DIR
        / "transformer_20layer_full_best.pt"

    )


    training_start = (
        time.time()
    )


    print(
        "\nStarting full 20-layer training..."
    )


    for epoch in range(

        1,

        EPOCHS + 1

    ):

        epoch_start = (
            time.time()
        )


        train_metrics = train_one_epoch(

            model=model,

            loader=train_loader,

            criterion=criterion,

            optimizer=optimizer,

            device=device,

            scaler=scaler,

            use_amp=use_amp

        )


        (

            val_metrics,

            _,

            _,

            _

        ) = evaluate(

            model=model,

            loader=val_loader,

            criterion=criterion,

            device=device,

            use_amp=use_amp

        )


        scheduler.step(
            val_metrics["f1"]
        )


        epoch_time = (

            time.time()
            - epoch_start

        )


        current_lr = (

            optimizer
            .param_groups[0][
                "lr"
            ]

        )


        print(
            f"\nEpoch "
            f"{epoch:02d}/{EPOCHS}"
        )


        print(

            "Train | "

            f"Loss: "
            f"{train_metrics['loss']:.4f} | "

            f"Acc: "
            f"{train_metrics['accuracy']:.4f} | "

            f"Precision: "
            f"{train_metrics['precision']:.4f} | "

            f"Recall: "
            f"{train_metrics['recall']:.4f} | "

            f"F1: "
            f"{train_metrics['f1']:.4f} | "

            f"AUC: "
            f"{train_metrics['auc']:.4f}"

        )


        print(

            "Val   | "

            f"Loss: "
            f"{val_metrics['loss']:.4f} | "

            f"Acc: "
            f"{val_metrics['accuracy']:.4f} | "

            f"Precision: "
            f"{val_metrics['precision']:.4f} | "

            f"Recall: "
            f"{val_metrics['recall']:.4f} | "

            f"F1: "
            f"{val_metrics['f1']:.4f} | "

            f"AUC: "
            f"{val_metrics['auc']:.4f}"

        )


        print(

            f"LR: "
            f"{current_lr:.6f} | "

            f"Epoch time: "
            f"{epoch_time:.1f} sec"

        )


        history.append({

            "epoch":
                epoch,

            "train_loss":
                train_metrics["loss"],

            "train_accuracy":
                train_metrics["accuracy"],

            "train_precision":
                train_metrics["precision"],

            "train_recall":
                train_metrics["recall"],

            "train_f1":
                train_metrics["f1"],

            "train_auc":
                train_metrics["auc"],

            "val_loss":
                val_metrics["loss"],

            "val_accuracy":
                val_metrics["accuracy"],

            "val_precision":
                val_metrics["precision"],

            "val_recall":
                val_metrics["recall"],

            "val_f1":
                val_metrics["f1"],

            "val_auc":
                val_metrics["auc"],

            "learning_rate":
                current_lr,

            "epoch_time_seconds":
                epoch_time

        })


        if (
            val_metrics["f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_metrics["f1"]
            )


            epochs_without_improvement = 0


            torch.save(

                {

                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_metrics":
                        val_metrics,

                    "class_weights":
                        class_weights
                        .detach()
                        .cpu()
                        .tolist(),

                    "total_parameters":
                        total_params,

                    "trainable_parameters":
                        trainable_params,

                    "model_config": {

                        "num_leads":
                            NUM_LEADS,

                        "signal_length":
                            SIGNAL_LENGTH,

                        "patch_size":
                            PATCH_SIZE,

                        "d_model":
                            D_MODEL,

                        "num_heads":
                            NUM_HEADS,

                        "num_layers":
                            NUM_LAYERS,

                        "dim_feedforward":
                            DIM_FEEDFORWARD,

                        "dropout":
                            DROPOUT,

                        "num_classes":
                            NUM_CLASSES
                    }

                },

                best_checkpoint_path

            )


            print(
                "Best 20-layer model saved."
            )


        else:

            epochs_without_improvement += 1


            print(

                "No validation F1 improvement "

                f"({epochs_without_improvement}/"

                f"{EARLY_STOPPING_PATIENCE})"

            )


        if (

            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE

        ):

            print(
                "\nEarly stopping triggered."
            )

            break


    # =====================================================
    # SAVE HISTORY
    # =====================================================

    history_df = pd.DataFrame(
        history
    )


    history_path = (

        RESULTS_DIR
        / "transformer_20layer_full_history.csv"

    )


    history_df.to_csv(

        history_path,

        index=False

    )


    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    print(
        "\nLoading best 20-layer checkpoint..."
    )


    checkpoint = torch.load(

        best_checkpoint_path,

        map_location=device,

        weights_only=False

    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    print(

        "Best validation F1:",

        f"{best_val_f1:.4f}"

    )


    # =====================================================
    # TEST
    # =====================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "20-LAYER TRANSFORMER FINAL TEST"
    )

    print(
        "=" * 75
    )


    (

        test_metrics,

        test_labels,

        test_predictions,

        test_probabilities

    ) = evaluate(

        model=model,

        loader=test_loader,

        criterion=criterion,

        device=device,

        use_amp=use_amp

    )


    print(

        f"Test Loss:      "
        f"{test_metrics['loss']:.4f}"

    )


    print(

        f"Test Accuracy:  "
        f"{test_metrics['accuracy']:.4f}"

    )


    print(

        f"Test Precision: "
        f"{test_metrics['precision']:.4f}"

    )


    print(

        f"Test Recall:    "
        f"{test_metrics['recall']:.4f}"

    )


    print(

        f"Test F1:        "
        f"{test_metrics['f1']:.4f}"

    )


    print(

        f"Test AUC:       "
        f"{test_metrics['auc']:.4f}"

    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(

        test_labels,

        test_predictions

    )


    print(
        "\nConfusion Matrix:"
    )


    print(
        cm
    )


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    print(
        "\nClassification Report:"
    )


    print(

        classification_report(

            test_labels,

            test_predictions,

            target_names=[
                "Non_MI",
                "MI"
            ],

            digits=4,

            zero_division=0

        )

    )


    # =====================================================
    # SAVE PREDICTIONS
    # =====================================================

    predictions_df = pd.DataFrame({

        "true_label":
            test_labels,

        "predicted_label":
            test_predictions,

        "mi_probability":
            test_probabilities

    })


    predictions_path = (

        RESULTS_DIR
        / "transformer_20layer_full_test_predictions.csv"

    )


    predictions_df.to_csv(

        predictions_path,

        index=False

    )


    # =====================================================
    # FINAL RESULTS
    # =====================================================

    total_training_time = (

        time.time()
        - training_start

    )


    results = {

        "experiment":
            "20-layer full Transformer",

        "frozen_encoder_layers":
            0,

        "num_encoder_layers":
            NUM_LAYERS,

        "total_parameters":
            total_params,

        "trainable_parameters":
            trainable_params,

        "batch_size":
            BATCH_SIZE,

        "train_samples":
            len(train_dataset),

        "validation_samples":
            len(val_dataset),

        "test_samples":
            len(test_dataset),

        "best_validation_f1":
            best_val_f1,

        "test_loss":
            test_metrics[
                "loss"
            ],

        "test_accuracy":
            test_metrics[
                "accuracy"
            ],

        "test_precision":
            test_metrics[
                "precision"
            ],

        "test_recall":
            test_metrics[
                "recall"
            ],

        "test_f1":
            test_metrics[
                "f1"
            ],

        "test_auc":
            test_metrics[
                "auc"
            ],

        "confusion_matrix":
            cm.tolist(),

        "epochs_completed":
            len(history),

        "training_time_seconds":
            total_training_time,

        "training_time_minutes":
            (
                total_training_time
                / 60
            )

    }


    results_path = (

        RESULTS_DIR
        / "transformer_20layer_full_results.json"

    )


    with open(

        results_path,

        "w"

    ) as f:

        json.dump(

            results,

            f,

            indent=4

        )


    # =====================================================
    # SUMMARY
    # =====================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "20-LAYER EXPERIMENT COMPLETE"
    )

    print(
        "=" * 75
    )


    print(
        "\nEncoder layers:",
        NUM_LAYERS
    )


    print(
        "Frozen encoder layers:",
        0
    )


    print(
        "Total parameters:",
        f"{total_params:,}"
    )


    print(
        "Trainable parameters:",
        f"{trainable_params:,}"
    )


    print(

        f"Total training time: "
        f"{total_training_time / 60:.2f} minutes"

    )


    print(
        "\nCheckpoint:"
    )


    print(
        best_checkpoint_path
    )


    print(
        "\nHistory:"
    )


    print(
        history_path
    )


    print(
        "\nResults:"
    )


    print(
        results_path
    )


if __name__ == "__main__":

    main()