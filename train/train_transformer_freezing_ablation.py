from pathlib import Path
import sys
import random
import json
import time
import copy

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
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42


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
# EXPERIMENT CONFIGURATION
# ============================================================

BATCH_SIZE = 8

NUM_WORKERS = 0

FINE_TUNE_EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 3

GRADIENT_CLIP = 1.0


# ------------------------------------------------------------
# Number of EARLY Transformer encoder blocks to freeze
# ------------------------------------------------------------

FREEZE_SETTINGS = [
    0,
    4,
    8,
    12,
    16,
]


# ============================================================
# MODEL CONFIGURATION
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

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

ABLATION_RESULTS_DIR = (
    RESULTS_DIR
    / "transformer_freezing_ablation"
)

BASE_CHECKPOINT = (
    CHECKPOINT_DIR
    / "transformer_20layer_full_best.pt"
)


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
            "Training data must contain labels 0 and 1."
        )

    total = len(df)

    weight_non_mi = (
        total
        / (2 * counts.loc[0])
    )

    weight_mi = (
        total
        / (2 * counts.loc[1])
    )

    return torch.tensor(
        [
            weight_non_mi,
            weight_mi
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

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "auc": auc
    }


# ============================================================
# LOCATE TRANSFORMER ENCODER
# ============================================================

def find_transformer_encoder(model):

    """
    Finds the torch.nn.TransformerEncoder inside ECGTransformer.

    This makes the script work even if the encoder attribute name
    is transformer_encoder, encoder, etc.
    """

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
# FREEZE EARLY TRANSFORMER BLOCKS
# ============================================================

def freeze_encoder_layers(
    model,
    num_layers_to_freeze
):

    encoder_name, encoder = (
        find_transformer_encoder(model)
    )

    layers = encoder.layers

    total_layers = len(layers)

    if num_layers_to_freeze > total_layers:

        raise ValueError(
            f"Cannot freeze "
            f"{num_layers_to_freeze} layers. "
            f"Model only has "
            f"{total_layers} encoder layers."
        )


    # --------------------------------------------------------
    # First make EVERYTHING trainable.
    # --------------------------------------------------------

    for parameter in model.parameters():

        parameter.requires_grad = True


    # --------------------------------------------------------
    # Freeze only the first N Transformer encoder blocks.
    # --------------------------------------------------------

    for layer_index in range(
        num_layers_to_freeze
    ):

        for parameter in (
            layers[layer_index]
            .parameters()
        ):

            parameter.requires_grad = False


    print(
        f"Transformer encoder found: "
        f"{encoder_name}"
    )

    print(
        f"Total encoder layers: "
        f"{total_layers}"
    )

    print(
        f"Frozen encoder layers: "
        f"{num_layers_to_freeze}"
    )

    print(
        f"Trainable encoder layers: "
        f"{total_layers - num_layers_to_freeze}"
    )


    return total_layers


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen_parameters = (
        total_parameters
        - trainable_parameters
    )

    return (
        total_parameters,
        trainable_parameters,
        frozen_parameters
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

    model = model.to(device)

    return model


# ============================================================
# LOAD COMMON 20-LAYER CHECKPOINT
# ============================================================

def load_base_checkpoint(
    model,
    device
):

    if not BASE_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"\nBase checkpoint not found:\n"
            f"{BASE_CHECKPOINT}\n\n"
            f"Run the 20-layer full training first."
        )


    checkpoint = torch.load(

        BASE_CHECKPOINT,

        map_location=device,

        weights_only=False

    )


    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    print(
        "Loaded common checkpoint:"
    )

    print(
        BASE_CHECKPOINT
    )


    if "epoch" in checkpoint:

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"]
        )


    if "val_metrics" in checkpoint:

        print(
            "Checkpoint validation metrics:",
            checkpoint["val_metrics"]
        )


    return checkpoint


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

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

            logits = model(signals)

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
                p
                for p in model.parameters()
                if p.requires_grad
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


    metrics["loss"] = epoch_loss


    return metrics


# ============================================================
# EVALUATE
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


    loss_value = (
        running_loss
        / len(loader.dataset)
    )


    metrics = calculate_metrics(

        all_labels,

        all_predictions,

        all_probabilities

    )


    metrics["loss"] = loss_value


    return (
        metrics,
        all_labels,
        all_predictions,
        all_probabilities
    )


# ============================================================
# RUN ONE FREEZING EXPERIMENT
# ============================================================

def run_experiment(
    frozen_layers,
    train_loader,
    val_loader,
    test_loader,
    class_weights,
    device,
    use_amp
):

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        f"EXPERIMENT: FREEZE FIRST "
        f"{frozen_layers} ENCODER LAYERS"
    )

    print(
        "=" * 80
    )


    # --------------------------------------------------------
    # Reset seed before EVERY experiment
    # --------------------------------------------------------

    set_seed(
        RANDOM_SEED
    )


    # --------------------------------------------------------
    # Create same architecture
    # --------------------------------------------------------

    model = create_model(
        device
    )


    # --------------------------------------------------------
    # Load same starting checkpoint
    # --------------------------------------------------------

    load_base_checkpoint(
        model,
        device
    )


    # --------------------------------------------------------
    # Freeze required encoder blocks
    # --------------------------------------------------------

    total_encoder_layers = (
        freeze_encoder_layers(

            model,
            frozen_layers

        )
    )


    (
        total_parameters,
        trainable_parameters,
        frozen_parameters

    ) = count_parameters(model)


    trainable_percentage = (

        100
        * trainable_parameters
        / total_parameters

    )


    print(
        "\nParameter information"
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    print(
        "Frozen parameters:",
        f"{frozen_parameters:,}"
    )

    print(
        "Trainable percentage:",
        f"{trainable_percentage:.2f}%"
    )


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Optimizer sees TRAINABLE PARAMETERS ONLY
    # --------------------------------------------------------

    trainable_parameter_list = [

        parameter

        for parameter
        in model.parameters()

        if parameter.requires_grad

    ]


    optimizer = torch.optim.AdamW(

        trainable_parameter_list,

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    scheduler = (

        torch.optim.lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=1

        )

    )


    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=use_amp

    )


    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    experiment_name = (
        f"freeze_{frozen_layers}"
    )


    checkpoint_path = (

        CHECKPOINT_DIR
        / (
            "transformer_20layer_"
            f"{experiment_name}_best.pt"
        )

    )


    history_path = (

        ABLATION_RESULTS_DIR
        / (
            f"{experiment_name}_history.csv"
        )

    )


    predictions_path = (

        ABLATION_RESULTS_DIR
        / (
            f"{experiment_name}_"
            "test_predictions.csv"
        )

    )


    results_path = (

        ABLATION_RESULTS_DIR
        / (
            f"{experiment_name}_results.json"
        )

    )


    # --------------------------------------------------------
    # Fine-tuning
    # --------------------------------------------------------

    best_val_f1 = -1.0

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    training_start = time.time()


    for epoch in range(
        1,
        FINE_TUNE_EPOCHS + 1
    ):

        epoch_start = time.time()


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
            optimizer.param_groups[0]["lr"]
        )


        print(
            f"\nEpoch "
            f"{epoch:02d}/"
            f"{FINE_TUNE_EPOCHS}"
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

            "epoch": epoch,

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


        # ----------------------------------------------------
        # Save best validation F1
        # ----------------------------------------------------

        if (
            val_metrics["f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_metrics["f1"]
            )

            best_epoch = epoch

            epochs_without_improvement = 0


            torch.save(

                {

                    "epoch": epoch,

                    "frozen_encoder_layers":
                        frozen_layers,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_metrics":
                        val_metrics,

                    "total_parameters":
                        total_parameters,

                    "trainable_parameters":
                        trainable_parameters,

                    "frozen_parameters":
                        frozen_parameters

                },

                checkpoint_path

            )


            print(
                "Best model saved."
            )


        else:

            epochs_without_improvement += 1


            print(

                "No validation F1 improvement "

                f"({epochs_without_improvement}/"

                f"{EARLY_STOPPING_PATIENCE})"

            )


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (

            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE

        ):

            print(
                "Early stopping triggered."
            )

            break


    total_training_time = (
        time.time()
        - training_start
    )


    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        history_path,
        index=False
    )


    # --------------------------------------------------------
    # Load best fine-tuned checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device,

        weights_only=False

    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

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


    cm = confusion_matrix(

        test_labels,
        test_predictions

    )


    print(
        "\n"
        + "-" * 80
    )

    print(
        f"FINAL TEST — "
        f"FREEZE {frozen_layers}"
    )

    print(
        "-" * 80
    )


    print(
        f"Accuracy:  "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1:        "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"AUC:       "
        f"{test_metrics['auc']:.4f}"
    )


    print(
        "\nConfusion Matrix:"
    )

    print(cm)


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


    print(
        f"Training time: "
        f"{total_training_time / 60:.2f} min"
    )


    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({

        "true_label":
            test_labels,

        "predicted_label":
            test_predictions,

        "mi_probability":
            test_probabilities

    })


    prediction_df.to_csv(
        predictions_path,
        index=False
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {

        "experiment":
            experiment_name,

        "total_encoder_layers":
            total_encoder_layers,

        "frozen_encoder_layers":
            frozen_layers,

        "trainable_encoder_layers":
            (
                total_encoder_layers
                - frozen_layers
            ),

        "total_parameters":
            total_parameters,

        "trainable_parameters":
            trainable_parameters,

        "frozen_parameters":
            frozen_parameters,

        "trainable_percentage":
            trainable_percentage,

        "best_epoch":
            best_epoch,

        "epochs_completed":
            len(history),

        "best_validation_f1":
            best_val_f1,

        "test_loss":
            test_metrics["loss"],

        "test_accuracy":
            test_metrics["accuracy"],

        "test_precision":
            test_metrics["precision"],

        "test_recall":
            test_metrics["recall"],

        "test_f1":
            test_metrics["f1"],

        "test_auc":
            test_metrics["auc"],

        "confusion_matrix":
            cm.tolist(),

        "training_time_seconds":
            total_training_time,

        "training_time_minutes":
            (
                total_training_time
                / 60
            ),

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE

    }


    with open(
        results_path,
        "w"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )


    # --------------------------------------------------------
    # Free GPU memory before next experiment
    # --------------------------------------------------------

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


    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ABLATION_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

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


    required_files = [

        train_csv,
        val_csv,
        test_csv,
        BASE_CHECKPOINT

    ]


    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file:\n"
                f"{path}"
            )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"

    )


    use_amp = (
        device.type == "cuda"
    )


    print(
        "=" * 80
    )

    print(
        "TRANSFORMER LAYER-FREEZING ABLATION STUDY"
    )

    print(
        "=" * 80
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
        "Base checkpoint:"
    )

    print(
        BASE_CHECKPOINT
    )


    print(
        "\nFreeze settings:"
    )

    print(
        FREEZE_SETTINGS
    )


    print(
        "\nFine-tuning learning rate:",
        LEARNING_RATE
    )

    print(
        "Fine-tuning epochs:",
        FINE_TUNE_EPOCHS
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )


    # ========================================================
    # PREPROCESSING
    # ========================================================

    signal_transform = ECGSignalPreprocessor(

        normalize=True,

        clip_value=5.0

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


    test_dataset = PTBXLDataset(

        csv_file=test_csv,

        ptbxl_root=PTBXL_ROOT,

        transform=signal_transform

    )


    print(
        "\nDataset sizes:"
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


    # ========================================================
    # DATALOADERS
    # ========================================================

    # Fixed generator improves reproducibility.

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=(
            device.type == "cuda"
        ),

        generator=generator

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
    # RUN ALL EXPERIMENTS
    # ========================================================

    all_results = []


    overall_start = time.time()


    for frozen_layers in FREEZE_SETTINGS:

        result = run_experiment(

            frozen_layers=frozen_layers,

            train_loader=train_loader,

            val_loader=val_loader,

            test_loader=test_loader,

            class_weights=class_weights,

            device=device,

            use_amp=use_amp

        )


        all_results.append(
            result
        )


    overall_time = (
        time.time()
        - overall_start
    )


    # ========================================================
    # FINAL COMPARISON TABLE
    # ========================================================

    comparison_rows = []


    for result in all_results:

        comparison_rows.append({

            "Frozen_Layers":
                result[
                    "frozen_encoder_layers"
                ],

            "Trainable_Layers":
                result[
                    "trainable_encoder_layers"
                ],

            "Total_Params":
                result[
                    "total_parameters"
                ],

            "Trainable_Params":
                result[
                    "trainable_parameters"
                ],

            "Trainable_Percent":
                result[
                    "trainable_percentage"
                ],

            "Best_Epoch":
                result[
                    "best_epoch"
                ],

            "Validation_F1":
                result[
                    "best_validation_f1"
                ],

            "Accuracy":
                result[
                    "test_accuracy"
                ],

            "Precision":
                result[
                    "test_precision"
                ],

            "Recall":
                result[
                    "test_recall"
                ],

            "F1":
                result[
                    "test_f1"
                ],

            "AUC":
                result[
                    "test_auc"
                ],

            "Training_Time_Min":
                result[
                    "training_time_minutes"
                ]

        })


    comparison_df = pd.DataFrame(
        comparison_rows
    )


    comparison_path = (

        RESULTS_DIR
        / "transformer_freezing_comparison.csv"

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
        + "=" * 100
    )

    print(
        "FINAL TRANSFORMER FREEZING COMPARISON"
    )

    print(
        "=" * 100
    )


    display_columns = [

        "Frozen_Layers",

        "Trainable_Params",

        "Validation_F1",

        "Accuracy",

        "Precision",

        "Recall",

        "F1",

        "AUC",

        "Training_Time_Min"

    ]


    print(

        comparison_df[
            display_columns
        ].to_string(
            index=False
        )

    )


    print(
        "\nTotal ablation runtime:"
    )

    print(
        f"{overall_time / 60:.2f} minutes"
    )


    print(
        "\nFinal comparison CSV:"
    )

    print(
        comparison_path
    )


    print(
        "\nIndividual experiment results:"
    )

    print(
        ABLATION_RESULTS_DIR
    )


if __name__ == "__main__":

    main()