from pathlib import Path
import sys

import torch
import torch.nn as nn


# =========================================================
# Project path setup
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


from models.cnn import ECGResNet18
from models.transformer import ECGTransformer


# =========================================================
# Hybrid model
# =========================================================

class ECGHybridModel(nn.Module):
    """
    Hybrid ECG classifier.

    CNN branch:
        rendered ECG image
        -> trained ResNet18
        -> 512-dimensional feature

    Transformer branch:
        raw 12-lead ECG signal
        -> trained Transformer
        -> 64-dimensional CLS feature

    Fusion:
        512 + 64 = 576 dimensions
        -> MLP classifier
        -> MI / Non-MI
    """

    def __init__(
        self,
        cnn_checkpoint=None,
        transformer_checkpoint=None,
        num_classes=2,
        freeze_backbones=True
    ):
        super().__init__()

        # =================================================
        # CNN branch
        # =================================================

        # IMPORTANT:
        # Use exactly the same ECGResNet18 class that was
        # used during CNN training.
        self.cnn = ECGResNet18(
            num_classes=2,
            pretrained=False,
            dropout=0.3
        )

        # =================================================
        # Transformer branch
        # =================================================

        self.transformer = ECGTransformer(
            num_leads=12,
            signal_length=1000,
            patch_size=10,
            d_model=64,
            num_heads=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            num_classes=2
        )

        # =================================================
        # Load trained CNN checkpoint
        # =================================================

        if cnn_checkpoint is not None:

            cnn_checkpoint = Path(
                cnn_checkpoint
            )

            if not cnn_checkpoint.exists():
                raise FileNotFoundError(
                    f"CNN checkpoint not found: "
                    f"{cnn_checkpoint}"
                )

            checkpoint = torch.load(
                cnn_checkpoint,
                map_location="cpu",
                weights_only=False
            )

            if "model_state_dict" not in checkpoint:
                raise KeyError(
                    "CNN checkpoint does not contain "
                    "'model_state_dict'."
                )

            self.cnn.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ],
                strict=True
            )

            print(
                "Loaded trained CNN checkpoint."
            )

        # =================================================
        # Load trained Transformer checkpoint
        # =================================================

        if transformer_checkpoint is not None:

            transformer_checkpoint = Path(
                transformer_checkpoint
            )

            if not transformer_checkpoint.exists():
                raise FileNotFoundError(
                    f"Transformer checkpoint not found: "
                    f"{transformer_checkpoint}"
                )

            checkpoint = torch.load(
                transformer_checkpoint,
                map_location="cpu",
                weights_only=False
            )

            if "model_state_dict" not in checkpoint:
                raise KeyError(
                    "Transformer checkpoint does not contain "
                    "'model_state_dict'."
                )

            self.transformer.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ],
                strict=True
            )

            print(
                "Loaded trained Transformer checkpoint."
            )

        # =================================================
        # Convert CNN classifier into feature extractor
        # =================================================

        # Your ECGResNet18 contains:
        #
        # self.backbone = ResNet18(...)
        #
        # Therefore replace:
        # self.cnn.backbone.fc
        #
        # with Identity.
        #
        # ResNet18 will now output its 512-dimensional
        # feature vector instead of 2 logits.

        self.cnn.backbone.fc = nn.Identity()  # type: ignore[assignment]

        # =================================================
        # Feature dimensions
        # =================================================

        self.cnn_feature_dim = 512
        self.transformer_feature_dim = 64

        fusion_dim = (
            self.cnn_feature_dim
            + self.transformer_feature_dim
        )

        # =================================================
        # Fusion classifier
        # =================================================

        self.fusion_classifier = nn.Sequential(

            nn.Linear(
                fusion_dim,
                256
            ),

            nn.BatchNorm1d(
                256
            ),

            nn.GELU(),

            nn.Dropout(
                0.35
            ),

            nn.Linear(
                256,
                64
            ),

            nn.GELU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                64,
                num_classes
            )
        )

        # =================================================
        # Freeze pretrained branches
        # =================================================

        if freeze_backbones:

            for param in self.cnn.parameters():
                param.requires_grad = False

            for param in self.transformer.parameters():
                param.requires_grad = False

            print(
                "CNN and Transformer backbones frozen."
            )

        # Fusion classifier must always remain trainable
        for param in self.fusion_classifier.parameters():
            param.requires_grad = True

    # =====================================================
    # CNN feature extraction
    # =====================================================

    def extract_cnn_features(
        self,
        images
    ):
        """
        Returns:
            [B, 512]
        """

        features = self.cnn(
            images
        )

        if features.ndim != 2:
            raise ValueError(
                "Unexpected CNN feature shape: "
                f"{features.shape}"
            )

        if (
            features.shape[1]
            != self.cnn_feature_dim
        ):
            raise ValueError(
                "Expected CNN feature dimension "
                f"{self.cnn_feature_dim}, "
                f"got {features.shape[1]}"
            )

        return features

    # =====================================================
    # Transformer feature extraction
    # =====================================================

    def extract_transformer_features(
        self,
        signals
    ):
        """
        Extract the Transformer CLS representation.

        Input:
            [B, 12, 1000]

        Output:
            [B, 64]
        """

        if signals.ndim != 3:
            raise ValueError(
                "Expected Transformer input "
                "[B, 12, 1000], "
                f"got {signals.shape}"
            )

        # ---------------------------------------------
        # Patch embedding
        # ---------------------------------------------

        x = (
            self.transformer
            .patch_embedding(
                signals
            )
        )

        # [B, 64, 100]
        #
        # -> [B, 100, 64]

        x = x.transpose(
            1,
            2
        )

        batch_size = x.shape[0]

        # ---------------------------------------------
        # CLS token
        # ---------------------------------------------

        cls_token = (
            self.transformer
            .cls_token
            .expand(
                batch_size,
                -1,
                -1
            )
        )

        x = torch.cat(
            [
                cls_token,
                x
            ],
            dim=1
        )

        # ---------------------------------------------
        # Positional embedding
        # ---------------------------------------------

        x = (
            x
            + self.transformer
            .position_embedding
        )

        x = (
            self.transformer
            .embedding_dropout(
                x
            )
        )

        # ---------------------------------------------
        # Transformer encoder
        # ---------------------------------------------

        x = (
            self.transformer
            .transformer_encoder(
                x
            )
        )

        # ---------------------------------------------
        # CLS representation
        # ---------------------------------------------

        cls_output = x[:, 0]

        cls_output = (
            self.transformer
            .norm(
                cls_output
            )
        )

        if (
            cls_output.shape[1]
            != self.transformer_feature_dim
        ):
            raise ValueError(
                "Expected Transformer feature "
                f"dimension "
                f"{self.transformer_feature_dim}, "
                f"got {cls_output.shape[1]}"
            )

        return cls_output

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        images,
        signals
    ):

        cnn_features = (
            self.extract_cnn_features(
                images
            )
        )

        transformer_features = (
            self.extract_transformer_features(
                signals
            )
        )

        # ---------------------------------------------
        # Feature fusion
        # ---------------------------------------------

        fused_features = torch.cat(
            [
                cnn_features,
                transformer_features
            ],
            dim=1
        )

        # [B, 576]

        logits = (
            self.fusion_classifier(
                fused_features
            )
        )

        return logits


# =========================================================
# Standalone test
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("HYBRID MODEL ARCHITECTURE TEST")
    print("=" * 70)

    model = ECGHybridModel(
        cnn_checkpoint=None,
        transformer_checkpoint=None,
        num_classes=2,
        freeze_backbones=True
    )

    images = torch.randn(
        4,
        3,
        224,
        224
    )

    signals = torch.randn(
        4,
        12,
        1000
    )

    with torch.no_grad():

        cnn_features = (
            model.extract_cnn_features(
                images
            )
        )

        transformer_features = (
            model.extract_transformer_features(
                signals
            )
        )

        output = model(
            images,
            signals
        )

    print(
        "\nImage input:",
        images.shape
    )

    print(
        "Signal input:",
        signals.shape
    )

    print(
        "\nCNN feature shape:",
        cnn_features.shape
    )

    print(
        "Transformer feature shape:",
        transformer_features.shape
    )

    print(
        "Fused feature dimension:",
        (
            cnn_features.shape[1]
            + transformer_features.shape[1]
        )
    )

    print(
        "\nOutput:",
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

    frozen_params = (
        total_params
        - trainable_params
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
        f"{frozen_params:,}"
    )

    assert (
        output.shape
        == (4, 2)
    )

    assert (
        cnn_features.shape
        == (4, 512)
    )

    assert (
        transformer_features.shape
        == (4, 64)
    )

    print(
        "\nHYBRID ARCHITECTURE TEST PASSED"
    )