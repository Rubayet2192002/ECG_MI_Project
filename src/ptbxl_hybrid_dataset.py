from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb

from PIL import Image
from torch.utils.data import Dataset


class PTBXLHybridDataset(Dataset):
    """
    Returns:
        rendered ECG image
        raw/preprocessed 12-lead ECG signal
        label

    Image shape after transform:
        [3, 224, 224]

    Signal shape:
        [12, 1000]
    """

    REQUIRED_COLUMNS = {
        "ecg_id",
        "filename_lr",
        "label",
        "class_name"
    }

    def __init__(
        self,
        csv_file,
        ptbxl_root,
        image_dir,
        image_transform=None,
        signal_transform=None,
        expected_leads=12,
        expected_samples=1000
    ):
        self.csv_file = Path(csv_file)
        self.ptbxl_root = Path(ptbxl_root)
        self.image_dir = Path(image_dir)

        self.image_transform = image_transform
        self.signal_transform = signal_transform

        self.expected_leads = expected_leads
        self.expected_samples = expected_samples

        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"CSV not found: {self.csv_file}"
            )

        if not self.ptbxl_root.exists():
            raise FileNotFoundError(
                f"PTB-XL root not found: {self.ptbxl_root}"
            )

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        self.data = pd.read_csv(
            self.csv_file
        )

        missing = (
            self.REQUIRED_COLUMNS
            - set(self.data.columns)
        )

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        ecg_id = int(
            row["ecg_id"]
        )

        label = int(
            row["label"]
        )

        # =================================================
        # IMAGE
        # =================================================

        image_path = (
            self.image_dir
            / f"{ecg_id}.png"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image missing: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.image_transform is not None:
            image = self.image_transform(
                image
            )

        # =================================================
        # SIGNAL
        # =================================================

        record_path = (
            self.ptbxl_root
            / str(row["filename_lr"])
        )

        header_path = Path(
            str(record_path) + ".hea"
        )

        if not header_path.exists():
            raise FileNotFoundError(
                f"WFDB record missing: {record_path}"
            )

        signal, fields = wfdb.rdsamp(
            str(record_path)
        )

        signal = np.asarray(
            signal,
            dtype=np.float32
        ).T

        expected_shape = (
            self.expected_leads,
            self.expected_samples
        )

        if signal.shape != expected_shape:
            raise ValueError(
                f"Unexpected signal shape for ECG {ecg_id}: "
                f"{signal.shape}"
            )

        if self.signal_transform is not None:
            signal = self.signal_transform(
                signal
            )

        signal = torch.from_numpy(
            np.ascontiguousarray(
                signal,
                dtype=np.float32
            )
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, signal, label