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

from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# Project path
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

from src.ptbxl_image_dataset import PTBXLImageDataset
from models.cnn import ECGResNet18


# =========================================================
# Configuration
# =========================================================

RANDOM_SEED = 42

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_WORKERS = 0

EPOCHS = 15

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 4

GRADIENT_CLIP = 1.0


# =========================================================
# Reproducibility
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
# Class weights
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

    if len(counts) != 2:

        raise ValueError(
            "Expected exactly two classes."
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
# Metrics
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
# Train one epoch
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

    labels_all = []
    predictions_all = []
    probabilities_all = []


    for images, labels in loader:

        images = images.to(
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
                images
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
            * images.size(0)
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
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        predictions_all.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        probabilities_all.extend(
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
        labels_all,
        predictions_all,
        probabilities_all
    )

    metrics["loss"] = epoch_loss

    return metrics


# =========================================================
# Evaluation
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

    labels_all = []
    predictions_all = []
    probabilities_all = []


    for images, labels in loader:

        images = images.to(
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
                images
            )

            loss = criterion(
                logits,
                labels
            )


        running_loss += (
            loss.item()
            * images.size(0)
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
            labels.cpu()
            .numpy()
            .tolist()
        )

        predictions_all.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

        probabilities_all.extend(
            probabilities.cpu()
            .numpy()
            .tolist()
        )


    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )


    metrics = calculate_metrics(
        labels_all,
        predictions_all,
        probabilities_all
    )

    metrics["loss"] = epoch_loss


    return (
        metrics,
        labels_all,
        predictions_all,
        probabilities_all
    )


# =========================================================
# Verify images exist
# =========================================================

def verify_images(
    csv_path,
    image_dir
):

    df = pd.read_csv(
        csv_path
    )

    missing = []

    for ecg_id in df["ecg_id"]:

        image_path = (
            image_dir
            / f"{int(ecg_id)}.png"
        )

        if not image_path.exists():

            missing.append(
                int(ecg_id)
            )

    return missing


# =========================================================
# Main
# =========================================================

def main():

    set_seed(
        RANDOM_SEED
    )


    # =====================================================
    # Paths
    # =====================================================

    DEMO_SPLIT_DIR = (
        PROJECT_ROOT
        / "data"
        / "demo_splits"
    )

    IMAGE_DIR = (
        PROJECT_ROOT
        / "data"
        / "rendered_images"
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
        DEMO_SPLIT_DIR
        / "train.csv"
    )

    val_csv = (
        DEMO_SPLIT_DIR
        / "val.csv"
    )

    test_csv = (
        DEMO_SPLIT_DIR
        / "test.csv"
    )


    # =====================================================
    # Device
    # =====================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
    )


    print("=" * 70)

    print(
        "PTB-XL CNN TRAINING"
    )

    print("=" * 70)


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
            torch.cuda.get_device_properties(0)
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


    # =====================================================
    # Check images
    # =====================================================

    print(
        "\nChecking rendered images..."
    )


    train_missing = verify_images(
        train_csv,
        IMAGE_DIR
    )

    val_missing = verify_images(
        val_csv,
        IMAGE_DIR
    )

    test_missing = verify_images(
        test_csv,
        IMAGE_DIR
    )


    total_missing = (
        len(train_missing)
        + len(val_missing)
        + len(test_missing)
    )


    if total_missing > 0:

        print(
            "\nCNN CANNOT START YET."
        )

        print(
            "Missing train images:",
            len(train_missing)
        )

        print(
            "Missing validation images:",
            len(val_missing)
        )

        print(
            "Missing test images:",
            len(test_missing)
        )

        print(
            "\nResume rendering with:"
        )

        print(
            "python src/render_ecg_images.py"
        )

        return


    print(
        "All required images found."
    )


    # =====================================================
    # Transforms
    # =====================================================

    train_transform = transforms.Compose([
        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),

        transforms.RandomRotation(
            degrees=2
        ),

        transforms.ColorJitter(
            brightness=0.05,
            contrast=0.05
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])


    eval_transform = transforms.Compose([
        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])


    # =====================================================
    # Datasets
    # =====================================================

    train_dataset = PTBXLImageDataset(
        csv_file=train_csv,
        image_dir=IMAGE_DIR,
        transform=train_transform
    )


    val_dataset = PTBXLImageDataset(
        csv_file=val_csv,
        image_dir=IMAGE_DIR,
        transform=eval_transform
    )


    test_dataset = PTBXLImageDataset(
        csv_file=test_csv,
        image_dir=IMAGE_DIR,
        transform=eval_transform
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
    # DataLoaders
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
    # Model
    # =====================================================

    model = ECGResNet18(
        num_classes=2,
        pretrained=True,
        dropout=0.3
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
        f"\nTotal parameters: "
        f"{total_params:,}"
    )


    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )


    # =====================================================
    # Class weights
    # =====================================================

    class_weights = (
        calculate_class_weights(
            train_csv
        )
    )


    print(
        "\nClass weights:",
        class_weights.tolist()
    )


    class_weights = (
        class_weights.to(
            device
        )
    )


    # =====================================================
    # Loss
    # =====================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # =====================================================
    # Optimizer
    # =====================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # =====================================================
    # Scheduler
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
    # Training
    # =====================================================

    best_val_f1 = -1.0

    epochs_without_improvement = 0


    best_checkpoint_path = (
        CHECKPOINT_DIR
        / "cnn_demo_best.pt"
    )


    history = []


    training_start_time = (
        time.time()
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
            optimizer.param_groups[0][
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


        # -------------------------------------------------
        # History
        # -------------------------------------------------

        history.append({
            "epoch":
                epoch,

            "train_loss":
                train_metrics[
                    "loss"
                ],

            "train_accuracy":
                train_metrics[
                    "accuracy"
                ],

            "train_precision":
                train_metrics[
                    "precision"
                ],

            "train_recall":
                train_metrics[
                    "recall"
                ],

            "train_f1":
                train_metrics[
                    "f1"
                ],

            "train_auc":
                train_metrics[
                    "auc"
                ],

            "val_loss":
                val_metrics[
                    "loss"
                ],

            "val_accuracy":
                val_metrics[
                    "accuracy"
                ],

            "val_precision":
                val_metrics[
                    "precision"
                ],

            "val_recall":
                val_metrics[
                    "recall"
                ],

            "val_f1":
                val_metrics[
                    "f1"
                ],

            "val_auc":
                val_metrics[
                    "auc"
                ],

            "learning_rate":
                current_lr
        })


        # -------------------------------------------------
        # Best checkpoint
        # -------------------------------------------------

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
                        .tolist()
                },

                best_checkpoint_path
            )


            print(
                "Best model saved."
            )


        else:

            epochs_without_improvement += 1


            print(
                "No validation F1 "
                "improvement "
                f"({epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )


        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping "
                "triggered."
            )

            break


    # =====================================================
    # Save history
    # =====================================================

    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(
        RESULTS_DIR
        / "cnn_demo_history.csv",
        index=False
    )


    # =====================================================
    # Load best model
    # =====================================================

    print(
        "\nLoading best CNN checkpoint..."
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
    # Test
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL CNN TEST EVALUATION"
    )

    print(
        "=" * 70
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
    # Confusion matrix
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
    # Classification report
    # =====================================================

    print(
        "\nClassification Report:"
    )


    report = classification_report(
        test_labels,
        test_predictions,

        target_names=[
            "Non_MI",
            "MI"
        ],

        digits=4,

        zero_division=0
    )


    print(
        report
    )


    # =====================================================
    # Save predictions
    # =====================================================

    prediction_df = pd.DataFrame({
        "true_label":
            test_labels,

        "predicted_label":
            test_predictions,

        "mi_probability":
            test_probabilities
    })


    prediction_df.to_csv(
        RESULTS_DIR
        / "cnn_demo_test_predictions.csv",
        index=False
    )


    # =====================================================
    # Save results
    # =====================================================

    results = {
        "model":
            "ResNet18",

        "dataset":
            "PTB-XL rendered ECG demo subset",

        "train_samples":
            len(train_dataset),

        "validation_samples":
            len(val_dataset),

        "test_samples":
            len(test_dataset),

        "best_validation_f1":
            best_val_f1,

        "test_loss":
            test_metrics["loss"],

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
            cm.tolist()
    }


    with open(
        RESULTS_DIR
        / "cnn_demo_results.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )


    # =====================================================
    # Finish
    # =====================================================

    total_training_time = (
        time.time()
        - training_start_time
    )


    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIR
    )


    print(
        "\nBest checkpoint:"
    )

    print(
        best_checkpoint_path
    )


    print(
        f"\nTotal training time: "
        f"{total_training_time / 60:.2f} "
        f"minutes"
    )


    print(
        "\nCNN TRAINING COMPLETE"
    )


if __name__ == "__main__":
    main()
