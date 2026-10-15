import torch
import torch.nn as nn


class ECGTransformer(nn.Module):
    """
    Lightweight Transformer for binary MI classification
    using 12-lead PTB-XL ECG signals.

    Expected input:
        [batch_size, 12, 1000]

    Output:
        [batch_size, 2]
    """

    def __init__(
        self,
        num_leads=12,
        signal_length=1000,
        patch_size=10,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        num_classes=2
    ):
        super().__init__()

        if signal_length % patch_size != 0:
            raise ValueError(
                "signal_length must be divisible by patch_size"
            )

        self.num_leads = num_leads
        self.signal_length = signal_length
        self.patch_size = patch_size

        self.num_patches = signal_length // patch_size

        # -------------------------------------------------
        # Convert ECG signal into patch tokens
        # -------------------------------------------------

        self.patch_embedding = nn.Conv1d(
            in_channels=num_leads,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        # -------------------------------------------------
        # CLS token
        # -------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_model
            )
        )

        # -------------------------------------------------
        # Positional embedding
        # -------------------------------------------------

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches + 1,
                d_model
            )
        )

        self.embedding_dropout = nn.Dropout(
            dropout
        )

        # -------------------------------------------------
        # Transformer encoder
        # -------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # -------------------------------------------------
        # Final normalization
        # -------------------------------------------------

        self.norm = nn.LayerNorm(
            d_model
        )

        # -------------------------------------------------
        # Classification head
        # -------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                d_model,
                d_model
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model,
                num_classes
            )
        )

        self._initialize_weights()


    def _initialize_weights(self):

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02
        )


    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape "
                f"[B, 12, 1000], got {x.shape}"
            )

        if x.shape[1] != self.num_leads:
            raise ValueError(
                f"Expected {self.num_leads} leads, "
                f"got {x.shape[1]}"
            )

        if x.shape[2] != self.signal_length:
            raise ValueError(
                f"Expected signal length "
                f"{self.signal_length}, "
                f"got {x.shape[2]}"
            )

        # x:
        # [B, 12, 1000]

        x = self.patch_embedding(
            x
        )

        # [B, 64, 100]

        x = x.transpose(
            1,
            2
        )

        # [B, 100, 64]

        batch_size = x.shape[0]

        cls_token = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            [
                cls_token,
                x
            ],
            dim=1
        )

        # [B, 101, 64]

        x = (
            x
            + self.position_embedding
        )

        x = self.embedding_dropout(
            x
        )

        x = self.transformer_encoder(
            x
        )

        # CLS representation
        cls_output = x[:, 0]

        cls_output = self.norm(
            cls_output
        )

        logits = self.classifier(
            cls_output
        )

        return logits


# =========================================================
# Standalone test
# =========================================================

if __name__ == "__main__":

    model = ECGTransformer()

    dummy_input = torch.randn(
        8,
        12,
        1000
    )

    output = model(
        dummy_input
    )

    print(
        "Input shape:",
        dummy_input.shape
    )

    print(
        "Output shape:",
        output.shape
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
        "Total parameters:",
        f"{total_params:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_params:,}"
    )

