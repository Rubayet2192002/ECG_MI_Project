import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


class ECGResNet18(nn.Module):
    """
    ResNet18 CNN for binary MI classification
    using rendered 12-lead ECG images.

    Input:
        [B, 3, 224, 224]

    Output:
        [B, 2]
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        dropout=0.3
    ):
        super().__init__()

        # -------------------------------------------------
        # Load ResNet18
        # -------------------------------------------------

        if pretrained:

            try:
                weights = (
                    ResNet18_Weights.DEFAULT
                )

                self.backbone = resnet18(
                    weights=weights
                )

                print(
                    "Loaded pretrained "
                    "ResNet18 weights."
                )

            except Exception as e:

                print(
                    "WARNING: Could not load "
                    "pretrained weights."
                )

                print(
                    "Reason:",
                    str(e)
                )

                print(
                    "Using randomly initialized "
                    "ResNet18 instead."
                )

                self.backbone = resnet18(
                    weights=None
                )

        else:

            self.backbone = resnet18(
                weights=None
            )

        # -------------------------------------------------
        # Replace classifier
        # -------------------------------------------------

        in_features = (
            self.backbone.fc.in_features
        )

        # ResNet.fc is typed as a Linear in PyTorch's stubs, but
        # replacing it with a Sequential head is a standard pattern.
        self.backbone.fc = nn.Sequential(  # type: ignore[assignment]
            nn.Dropout(
                dropout
            ),

            nn.Linear(
                in_features,
                num_classes
            )
        )


    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":

    model = ECGResNet18(
        num_classes=2,
        pretrained=False
    )

    dummy_input = torch.randn(
        4,
        3,
        224,
        224
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