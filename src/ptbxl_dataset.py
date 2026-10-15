from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb


class PTBXLDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL records100 ECG signals.

    CSV must contain:
        ecg_id
        patient_id
        filename_lr
        label
        class_name

    Output:
        signal -> Tensor [12, 1000]
        label  -> scalar LongTensor
    """

    REQUIRED_COLUMNS = {
        "ecg_id",
        "patient_id",
        "filename_lr",
        "label",
        "class_name"
    }


    def __init__(
        self,
        csv_file,
        ptbxl_root,
        transform=None,
        expected_leads=12,
        expected_samples=1000
    ):
        self.csv_file = Path(csv_file)
        self.ptbxl_root = Path(ptbxl_root)

        self.transform = transform

        self.expected_leads = expected_leads
        self.expected_samples = expected_samples


        # -------------------------------------------------
        # Validate paths
        # -------------------------------------------------

        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_file}"
            )

        if not self.ptbxl_root.exists():
            raise FileNotFoundError(
                f"PTB-XL folder not found: {self.ptbxl_root}"
            )


        # -------------------------------------------------
        # Load CSV
        # -------------------------------------------------

        self.data = pd.read_csv(
            self.csv_file
        )


        # -------------------------------------------------
        # Check required columns
        # -------------------------------------------------

        missing_columns = (
            self.REQUIRED_COLUMNS
            - set(self.data.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing required CSV columns: "
                f"{sorted(missing_columns)}"
            )


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        row = self.data.iloc[idx]


        # -------------------------------------------------
        # Build WFDB record path
        # -------------------------------------------------

        record_path = (
            self.ptbxl_root
            / str(row["filename_lr"])
        )

        header_path = Path(
            str(record_path) + ".hea"
        )

        if not header_path.exists():
            raise FileNotFoundError(
                f"WFDB record not found for ECG "
                f"{row['ecg_id']}: {record_path}"
            )


        # -------------------------------------------------
        # Load ECG
        # -------------------------------------------------

        try:

            signal, fields = wfdb.rdsamp(
                str(record_path)
            )

        except Exception as e:

            raise RuntimeError(
                f"Failed loading ECG "
                f"{row['ecg_id']} from "
                f"{record_path}"
            ) from e


        # -------------------------------------------------
        # WFDB format:
        #
        # [samples, leads]
        #
        # records100:
        # [1000, 12]
        #
        # Convert to:
        # [12, 1000]
        # -------------------------------------------------

        signal = np.asarray(
            signal,
            dtype=np.float32
        ).T


        # -------------------------------------------------
        # Shape validation
        # -------------------------------------------------

        expected_shape = (
            self.expected_leads,
            self.expected_samples
        )

        if signal.shape != expected_shape:

            raise ValueError(
                f"Unexpected ECG shape for "
                f"ecg_id={row['ecg_id']}. "
                f"Expected {expected_shape}, "
                f"got {signal.shape}"
            )


        # -------------------------------------------------
        # Apply preprocessing
        # -------------------------------------------------

        if self.transform is not None:

            signal = self.transform(
                signal
            )


        # -------------------------------------------------
        # Final safety check
        # -------------------------------------------------

        if not np.isfinite(signal).all():

            raise ValueError(
                f"Non-finite values remain after "
                f"preprocessing for "
                f"ecg_id={row['ecg_id']}"
            )


        # -------------------------------------------------
        # Convert to PyTorch
        # -------------------------------------------------

        signal = torch.from_numpy(
            np.ascontiguousarray(
                signal,
                dtype=np.float32
            )
        )

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )


        return signal, label