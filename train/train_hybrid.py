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


from src.ptbxl_hybrid_dataset import PTBXLHybridDataset
from src.signal_preprocessing import ECGSignalPreprocessor
from models.hybrid import ECGHybridModel


# =========================================================
# Configuration
# =========================================================

RANDOM_SEED = 42

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_WORKERS = 0

EPOCHS = 15

LEARNING_RATE = 5e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 4

GRADIENT_CLIP = 1.0


# =========================================================
# Seed
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

    total = len(df)

    weights = []

    for label in [
        0,
        1
    ]:

        weight = (
            total
            / (
                2
                * counts.loc[label]
            )
        )

        weights.append(
            weight
        )

    return torch.tensor(
        weights,
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

    return {

        "accuracy":
            accuracy_score(
                labels,
                predictions
            ),

        "precision":
            precision_score(
                labels,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                labels,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                labels,
                predictions,
                zero_division=0
            ),

        "auc":
            roc_auc_score(
                labels,
                probabilities
            )
    }


# =========================================================
# Train epoch
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


    for (
        images,
        signals,
        labels
    ) in loader:

        images = images.to(
            device
        )

        signals = signals.to(
            device
        )

        labels = labels.to(
            device
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
                images,
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
            * labels.size(0)
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
            .tolist()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .tolist()
        )

        all_probabilities.extend(
            probabilities.detach()
            .cpu()
            .tolist()
        )


    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_probabilities
    )


    metrics["loss"] = (
        running_loss
        / len(loader.dataset)
    )


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

    all_labels = []
    all_predictions = []
    all_probabilities = []


    for (
        images,
        signals,
        labels
    ) in loader:

        images = images.to(
            device
        )

        signals = signals.to(
            device
        )

        labels = labels.to(
            device
        )


        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            logits = model(
                images,
                signals
            )

            loss = criterion(
                logits,
                labels
            )


        running_loss += (
            loss.item()
            * labels.size(0)
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
            labels.cpu()
            .tolist()
        )

        all_predictions.extend(
            predictions.cpu()
            .tolist()
        )

        all_probabilities.extend(
            probabilities.cpu()
            .tolist()
        )


    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_probabilities
    )


    metrics["loss"] = (
        running_loss
        / len(loader.dataset)
    )


    return (
        metrics,
        all_labels,
        all_predictions,
        all_probabilities
    )


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

    PTBXL_ROOT = (
        PROJECT_ROOT
        / "PTB_XL"
    )

    IMAGE_DIR = (
        PROJECT_ROOT
        / "data"
        / "rendered_images"
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


    CNN_CHECKPOINT = (
        CHECKPOINT_DIR
        / "cnn_demo_best.pt"
    )

    TRANSFORMER_CHECKPOINT = (
        CHECKPOINT_DIR
        / "transformer_demo_best.pt"
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


    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # =====================================================
    # Check pretrained models
    # =====================================================

    if not CNN_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"CNN checkpoint missing: "
            f"{CNN_CHECKPOINT}"
        )


    if not TRANSFORMER_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Transformer checkpoint missing: "
            f"{TRANSFORMER_CHECKPOINT}"
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
    print("PTB-XL HYBRID CNN + TRANSFORMER")
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

    print(
        "Mixed precision:",
        use_amp
    )


    # =====================================================
    # Image transforms
    # =====================================================

    train_image_transform = transforms.Compose([

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


    eval_image_transform = transforms.Compose([

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
    # Signal transform
    # =====================================================

    signal_transform = ECGSignalPreprocessor(
        normalize=True,
        clip_value=5.0
    )


    # =====================================================
    # Dataset
    # =====================================================

    train_dataset = PTBXLHybridDataset(
        csv_file=train_csv,
        ptbxl_root=PTBXL_ROOT,
        image_dir=IMAGE_DIR,
        image_transform=train_image_transform,
        signal_transform=signal_transform
    )


    val_dataset = PTBXLHybridDataset(
        csv_file=val_csv,
        ptbxl_root=PTBXL_ROOT,
        image_dir=IMAGE_DIR,
        image_transform=eval_image_transform,
        signal_transform=signal_transform
    )


    test_dataset = PTBXLHybridDataset(
        csv_file=test_csv,
        ptbxl_root=PTBXL_ROOT,
        image_dir=IMAGE_DIR,
        image_transform=eval_image_transform,
        signal_transform=signal_transform
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
    # DataLoader
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
    # Hybrid model
    # =====================================================

    model = ECGHybridModel(
        cnn_checkpoint=CNN_CHECKPOINT,
        transformer_checkpoint=TRANSFORMER_CHECKPOINT,
        freeze_backbones=True
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


    # =====================================================
    # Loss
    # =====================================================

    class_weights = (
        calculate_class_weights(
            train_csv
        )
        .to(device)
    )


    print(
        "Class weights:",
        class_weights.tolist()
    )


    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # =====================================================
    # Optimizer
    # =====================================================

    optimizer = torch.optim.AdamW(
        filter(
            lambda p:
                p.requires_grad,
            model.parameters()
        ),

        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )


    # =====================================================
    # Train
    # =====================================================

    best_val_f1 = -1.0

    no_improvement = 0

    history = []


    BEST_PATH = (
        CHECKPOINT_DIR
        / "hybrid_demo_best.pt"
    )


    start_time = time.time()


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()


        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp
        )


        (
            val_metrics,
            _,
            _,
            _
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp
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
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"Precision: {train_metrics['precision']:.4f} | "
            f"Recall: {train_metrics['recall']:.4f} | "
            f"F1: {train_metrics['f1']:.4f} | "
            f"AUC: {train_metrics['auc']:.4f}"
        )


        print(
            "Val   | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f}"
        )


        print(
            f"LR: {current_lr:.6f} | "
            f"Epoch time: {epoch_time:.1f} sec"
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
                val_metrics["auc"]
        })


        if (
            val_metrics["f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_metrics["f1"]
            )

            no_improvement = 0


            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "val_metrics":
                        val_metrics
                },

                BEST_PATH
            )


            print(
                "Best hybrid model saved."
            )


        else:

            no_improvement += 1

            print(
                "No validation F1 improvement "
                f"({no_improvement}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )


        if (
            no_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break


    # =====================================================
    # Save history
    # =====================================================

    pd.DataFrame(
        history
    ).to_csv(
        RESULTS_DIR
        / "hybrid_demo_history.csv",
        index=False
    )


    # =====================================================
    # Load best model
    # =====================================================

    checkpoint = torch.load(
        BEST_PATH,
        map_location=device,
        weights_only=False
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    print(
        "\nBest validation F1:",
        f"{best_val_f1:.4f}"
    )


    # =====================================================
    # Final test
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL HYBRID TEST EVALUATION"
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
        model,
        test_loader,
        criterion,
        device,
        use_amp
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
    # Save test predictions
    # =====================================================

    pd.DataFrame({

        "true_label":
            test_labels,

        "predicted_label":
            test_predictions,

        "mi_probability":
            test_probabilities

    }).to_csv(
        RESULTS_DIR
        / "hybrid_demo_test_predictions.csv",
        index=False
    )


    # =====================================================
    # Save results
    # =====================================================

    results = {

        "model":
            "CNN + Transformer Hybrid",

        "dataset":
            "PTB-XL demo subset",

        "train_samples":
            len(train_dataset),

        "validation_samples":
            len(val_dataset),

        "test_samples":
            len(test_dataset),

        "best_validation_f1":
            best_val_f1,

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
        / "hybrid_demo_results.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )


    elapsed = (
        time.time()
        - start_time
    )


    print(
        f"\nTotal training time: "
        f"{elapsed / 60:.2f} minutes"
    )


    print(
        "\nHYBRID TRAINING COMPLETE"
    )


if __name__ == "__main__":
    main()
    