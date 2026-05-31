"""CNN training: AdamW + CrossEntropy(smoothing) + CosineAnnealing + Early Stop"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from app.cnn.model import create_model, count_parameters
from app.cnn.dataset import create_dataloaders
from app.common.logger import logger


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_norm: float = 1.0,
) -> float:
    model.train()
    running_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


def train(
    data_root: str,
    batch_size: int = 32,
    max_epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.1,
    early_stop_patience: int = 10,
    grad_max_norm: float = 1.0,
) -> str:
    """Train on Oxford-IIIT Pet Dataset.

    Args:
        data_root: Path to dataset root directory.
        batch_size: Mini-batch size.
        max_epochs: Maximum training epochs.
        lr: Initial learning rate for AdamW.
        weight_decay: Decoupled weight decay for AdamW.
        label_smoothing: Label smoothing factor (0 = none).
        early_stop_patience: Epochs without val_acc improvement before stopping.
        grad_max_norm: Gradient clipping max norm.

    Returns:
        Path to saved best model weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        data_root, batch_size
    )

    model = create_model(num_classes=num_classes).to(device)
    count_parameters(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    save_dir = os.path.join(os.path.dirname(__file__), "../../resources/models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "pet_cnn.pth")

    logger.info(
        "Training: max_epochs=%d, early_stop=%d, lr=%.4f, wd=%.0e, "
        "smoothing=%.1f, grad_clip=%.1f, batch=%d",
        max_epochs, early_stop_patience, lr, weight_decay,
        label_smoothing, grad_max_norm, batch_size,
    )

    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, grad_max_norm
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.perf_counter() - t0
        logger.info(
            "Epoch %2d/%d | train_loss: %.4f | val_loss: %.4f | val_acc: %.2f%% | lr: %.6f | %.1fs",
            epoch, max_epochs, train_loss, val_loss, val_acc * 100, current_lr, elapsed,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info("  -> saved best (val_acc=%.2f%%)", val_acc * 100)
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            logger.info(
                "Early stop @ epoch %d (best: epoch %d, val_acc=%.2f%%)",
                epoch, best_epoch, best_val_acc * 100,
            )
            break

    # Final test evaluation
    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True)
    )
    test_loss, test_acc = evaluate(model, test_loader, device)
    logger.info("Test: loss=%.4f, accuracy=%.2f%%", test_loss, test_acc * 100)

    target_met = test_acc >= 0.85
    logger.info(
        "Complete. Best val_acc=%.2f%% (epoch %d). Test_acc=%.2f%%. 85%%: %s",
        best_val_acc * 100, best_epoch, test_acc * 100,
        "PASS" if target_met else "NOT MET",
    )

    return save_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.cnn.train <data_root>")
        sys.exit(1)
    train(sys.argv[1])
