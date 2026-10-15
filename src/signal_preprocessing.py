import numpy as np


class ECGSignalPreprocessor:
    """
    Preprocessing pipeline for PTB-XL ECG signals.

    Expected input shape:
        [leads, samples]

    For records100:
        [12, 1000]
    """

    def __init__(
        self,
        normalize=True,
        clip_value=5.0,
        eps=1e-8
    ):
        self.normalize = normalize
        self.clip_value = clip_value
        self.eps = eps


    def __call__(self, signal):
        signal = np.asarray(
            signal,
            dtype=np.float32
        )

        # -------------------------------------------------
        # Validate dimensions
        # -------------------------------------------------

        if signal.ndim != 2:
            raise ValueError(
                f"Expected 2D ECG signal, got shape {signal.shape}"
            )


        # -------------------------------------------------
        # Replace invalid values
        # -------------------------------------------------

        signal = np.nan_to_num(
            signal,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


        # -------------------------------------------------
        # Per-lead Z-score normalization
        # -------------------------------------------------

        if self.normalize:

            mean = signal.mean(
                axis=1,
                keepdims=True
            )

            std = signal.std(
                axis=1,
                keepdims=True
            )

            # Avoid division by zero
            std = np.where(
                std < self.eps,
                1.0,
                std
            )

            signal = (
                signal - mean
            ) / std


        # -------------------------------------------------
        # Clip extreme normalized values
        # -------------------------------------------------

        if self.clip_value is not None:

            signal = np.clip(
                signal,
                -self.clip_value,
                self.clip_value
            )


        # Ensure contiguous float32 output
        signal = np.ascontiguousarray(
            signal,
            dtype=np.float32
        )

        return signal