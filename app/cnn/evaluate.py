"""Standalone model evaluation on test set — includes confusion matrix."""
import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

from app.cnn.dataset import create_dataloaders
from app.cnn.model import create_model
from app.common.logger import logger


@torch.no_grad()
def _evaluate(
    model: nn.Module, loader, device: torch.device
) -> tuple[float, float, list[int], list[int]]:
    """Run inference on the entire loader.

    Returns:
        (avg_loss, accuracy, all_targets, all_preds)
    """
    model.eval()
    running_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    total = len(all_targets)
    correct = sum(1 for t, p in zip(all_targets, all_preds, strict=True) if t == p)
    return running_loss / total, correct / total, all_targets, all_preds


def _plot_confusion_matrix(
    targets: list[int],
    preds: list[int],
    num_classes: int,
    save_path: str,
) -> None:
    """Build and save a confusion-matrix heatmap as PNG."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds, strict=True):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        square=True,
        linewidths=0.1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Oxford-IIIT Pet (37 breeds)", fontsize=16, pad=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def evaluate_model(
    data_root: str,
    weights_path: str | None = None,
    output_dir: str | None = None,
) -> tuple[float, float]:
    """Load trained model, evaluate on test set, and save confusion matrix.

    Args:
        data_root: Path to the Oxford-IIIT Pet dataset root.
        weights_path: Path to trained .pth file. If None, defaults to
            settings.model_weights_path.
        output_dir: Directory for the confusion matrix PNG. If None,
            defaults to settings.confusion_matrix_dir.

    Returns:
        (test_loss, test_accuracy)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if weights_path is None:
        from app.common.config import settings
        weights_path = settings.model_weights_path
    weights_path = os.path.normpath(weights_path)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    _, _, test_loader, num_classes = create_dataloaders(data_root, batch_size=32)
    model = create_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    test_loss, test_acc, all_targets, all_preds = _evaluate(model, test_loader, device)
    logger.info("Test: loss=%.4f, accuracy=%.2f%%", test_loss, test_acc * 100)

    # ── confusion matrix ──────────────────────────────────────────────
    if output_dir is None:
        from app.common.config import settings
        output_dir = settings.confusion_matrix_dir
    cm_path = os.path.normpath(os.path.join(output_dir, "confusion_matrix.png"))
    _plot_confusion_matrix(all_targets, all_preds, num_classes, cm_path)

    return test_loss, test_acc
