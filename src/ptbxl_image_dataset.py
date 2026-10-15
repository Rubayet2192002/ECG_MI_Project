from pathlib import Path

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset


class PTBXLImageDataset(Dataset):
    """
    PTB-XL rendered ECG image dataset.

    Expected CSV columns:
        ecg_id
        label
        class_name

    Rendered image:
        data/rendered_images/{ecg_id}.png
    """

    REQUIRED_COLUMNS = {
        "ecg_id",
        "label",
        "class_name"
    }

    def __init__(
        self,
        csv_file,
        image_dir,
        transform=None
    ):
        self.csv_file = Path(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform

        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_file}"
            )

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        self.data = pd.read_csv(
            self.csv_file
        )

        missing_columns = (
            self.REQUIRED_COLUMNS
            - set(self.data.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing_columns)}"
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

        image_path = (
            self.image_dir
            / f"{ecg_id}.png"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Rendered ECG image not found: "
                f"{image_path}"
            )

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load ECG image "
                f"{image_path}"
            ) from e

        if self.transform is not None:
            image = self.transform(
                image
            )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label