import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from sklearn.model_selection import train_test_split

from app.common.logger import logger

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class _TransformSubset(Dataset):
    """Applies a transform on top of a Subset."""

    def __init__(self, subset: Subset, transform: transforms.Compose | None = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int):
        img, target = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.subset)


def create_dataloaders(
    data_root: str, batch_size: int = 32
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Create stratified train/val/test DataLoaders for Oxford-IIIT Pet Dataset.

    Split: 70% train / 20% validation / 10% test, stratified by breed.
    Uses data augmentation on training set only.

    Args:
        data_root: Root directory for the dataset (auto-downloads on first use).
        batch_size: Batch size for all DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, num_classes)
    """
    # --- data augmentation (training only) ---
    train_augment = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    # --- preprocessing (validation & test) ---
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    num_classes = 37

    # --- load dataset (triggers download on first use) ---
    logger.info(f"Loading Oxford-IIIT Pet Dataset from {data_root} ...")
    full_dataset = OxfordIIITPet(root=data_root, download=False)

    # OxfordIIITPet uses 1-indexed class labels (1..37); convert to 0-indexed
    indices = list(range(len(full_dataset)))
    targets = []
    for i in range(len(full_dataset)):
        _, label = full_dataset[i]
        targets.append(label - 1)

    logger.info(
        "Dataset loaded: %d images, %d breeds", len(full_dataset), num_classes
    )

    # --- stratified 70/20/10 split ---
    train_idx, val_test_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=SEED,
        stratify=targets,
    )
    # targets for the 30% pool, needed for the second stratified split
    val_test_targets = [targets[i] for i in val_test_idx]
    val_idx, test_idx = train_test_split(
        val_test_idx,
        test_size=1.0 / 3.0,  # 10% of total = 1/3 of the 30% pool
        random_state=SEED,
        stratify=val_test_targets,
    )

    logger.info(
        "Split: train=%d  val=%d  test=%d",
        len(train_idx), len(val_idx), len(test_idx),
    )

    # --- build DataLoaders (num_workers=0 for Windows compatibility) ---
    train_dataset = _TransformSubset(
        Subset(full_dataset, train_idx), train_augment
    )
    val_dataset = _TransformSubset(
        Subset(full_dataset, val_idx), preprocess
    )
    test_dataset = _TransformSubset(
        Subset(full_dataset, test_idx), preprocess
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader, num_classes
