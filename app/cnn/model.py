"""CNN model: EfficientNet-B0 backbone (ImageNet pretrained) + custom classifier."""
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from app.common.logger import logger


def create_model(num_classes: int = 37) -> nn.Module:
    """EfficientNet-B0 with ImageNet pretrained weights, custom classification head.

    EfficientNet-B0 (~5.3M params) is chosen over ResNet50 (~24M) for:
    - 4-5x faster CPU training (lower FLOPs)
    - Lower memory footprint
    - Comparable accuracy on fine-grained classification

    Input:  (B, 3, 224, 224)
    Output: (B, num_classes) — raw logits
    """
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier: 1280 -> 512 -> num_classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )

    logger.info("EfficientNet-B0 created (pretrained ImageNet, %d classes)", num_classes)
    return model  # type: ignore[no-any-return]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters:    %d", total)
    logger.info("Trainable parameters: %d", trainable)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable


if __name__ == "__main__":
    import torch

    from app.common.logger import setup_logging
    setup_logging()

    m = create_model()
    count_parameters(m)

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = m(dummy)
    assert out.shape == (1, 37), f"Unexpected output shape: {out.shape}"
    logger.info("Forward pass OK: (1,3,224,224) -> %s", out.shape)
    print("All self-tests passed.")
