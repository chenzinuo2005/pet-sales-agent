"""CNN model: ResNet50 backbone (ImageNet pretrained) + custom classifier."""
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from app.common.logger import logger


def create_model(num_classes: int = 37) -> nn.Module:
    """ResNet50 with ImageNet pretrained weights, custom classification head.

    Input:  (B, 3, 224, 224)
    Output: (B, num_classes) — raw logits
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Replace final FC: 2048 -> 256 -> num_classes
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )

    logger.info("ResNet50 created (pretrained ImageNet, %d classes)", num_classes)
    return model


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
