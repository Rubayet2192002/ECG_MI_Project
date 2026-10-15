from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.ptbxl_dataset import PTBXLDataset
from src.signal_preprocessing import ECGSignalPreprocessor


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

PTBXL_ROOT = (
    PROJECT_ROOT
    / "PTB_XL"
)

TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.csv"
)

VAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "val.csv"
)

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test.csv"
)


# =========================================================
# Preprocessing
# =========================================================

signal_transform = ECGSignalPreprocessor(
    normalize=True,
    clip_value=5.0
)


# =========================================================
# Datasets
# =========================================================

train_dataset = PTBXLDataset(
    csv_file=TRAIN_CSV,
    ptbxl_root=PTBXL_ROOT,
    transform=signal_transform
)

val_dataset = PTBXLDataset(
    csv_file=VAL_CSV,
    ptbxl_root=PTBXL_ROOT,
    transform=signal_transform
)

test_dataset = PTBXLDataset(
    csv_file=TEST_CSV,
    ptbxl_root=PTBXL_ROOT,
    transform=signal_transform
)


# =========================================================
# Dataset size check
# =========================================================

print("=" * 60)
print("DATASET SIZE")
print("=" * 60)

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


# =========================================================
# Single sample check
# =========================================================

signal, label = train_dataset[0]

print("\n" + "=" * 60)
print("SINGLE SAMPLE")
print("=" * 60)

print(
    "Signal shape:",
    signal.shape
)

print(
    "Label:",
    label
)

print(
    "Signal dtype:",
    signal.dtype
)

print(
    "Signal minimum:",
    signal.min().item()
)

print(
    "Signal maximum:",
    signal.max().item()
)

print(
    "Signal mean:",
    signal.mean().item()
)

print(
    "Signal std:",
    signal.std().item()
)

print(
    "Contains NaN:",
    torch.isnan(signal).any().item()
)

print(
    "Contains Inf:",
    torch.isinf(signal).any().item()
)


# =========================================================
# Per-lead statistics
# =========================================================

print("\n" + "=" * 60)
print("PER-LEAD STATISTICS")
print("=" * 60)

for lead_index in range(signal.shape[0]):

    lead = signal[
        lead_index
    ]

    print(
        f"Lead {lead_index + 1:02d} | "
        f"Mean: {lead.mean().item():.4f} | "
        f"Std: {lead.std().item():.4f} | "
        f"Min: {lead.min().item():.4f} | "
        f"Max: {lead.max().item():.4f}"
    )


# =========================================================
# DataLoader
# =========================================================

BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


signals, labels = next(
    iter(train_loader)
)


print("\n" + "=" * 60)
print("BATCH")
print("=" * 60)

print(
    "Signals shape:",
    signals.shape
)

print(
    "Labels shape:",
    labels.shape
)

print(
    "Signals dtype:",
    signals.dtype
)

print(
    "Labels dtype:",
    labels.dtype
)

print(
    "Labels:",
    labels
)

print(
    "Batch contains NaN:",
    torch.isnan(signals).any().item()
)

print(
    "Batch contains Inf:",
    torch.isinf(signals).any().item()
)


# =========================================================
# Assertions
# =========================================================

assert len(train_dataset) == 17418
assert len(val_dataset) == 2183
assert len(test_dataset) == 2198

assert signal.shape == (
    12,
    1000
)

assert signals.shape == (
    BATCH_SIZE,
    12,
    1000
)

assert labels.shape == (
    BATCH_SIZE,
)

assert torch.isfinite(
    signal
).all()

assert torch.isfinite(
    signals
).all()


print("\n" + "=" * 60)
print("ALL PTB-XL DATA PIPELINE TESTS PASSED")
print("=" * 60)