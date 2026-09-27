from pathlib import Path
from typing import Tuple

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


IMAGE_SIZE = 224


def build_transforms(image_size: int = IMAGE_SIZE):
    """Create training and evaluation transforms.

    Training gets random augmentation so the model does not simply
    memorize the exact orientation/background of the training images.
    Validation/test transforms are deterministic.
    """

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.12,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


def build_loaders(
    dataset_root: str = "data",
    train_dir: str | None = None,
    test_dir: str | None = None,
    val_dir: str | None = None,
    image_size: int = IMAGE_SIZE,
    val_fraction: float = 0.2,
    batch_size: int = 16,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Load a folder-based dataset and create train/val/test loaders.

    A dataset root may contain ``train``, ``valid`` (or ``val``), and
    ``test`` directories. If no validation directory exists, a deterministic
    validation split is created from ``train``.
    """

    root = Path(dataset_root)
    train_path = Path(train_dir) if train_dir else root / "train"
    test_path = Path(test_dir) if test_dir else root / "test"
    if val_dir:
        val_path = Path(val_dir)
    elif (root / "valid").is_dir():
        val_path = root / "valid"
    elif (root / "val").is_dir():
        val_path = root / "val"
    else:
        val_path = None

    train_transform, eval_transform = build_transforms(image_size)

    full_train = datasets.ImageFolder(str(train_path), transform=train_transform)
    test_dataset = datasets.ImageFolder(str(test_path), transform=eval_transform)

    if val_path is not None:
        if full_train.classes != datasets.ImageFolder(str(val_path)).classes:
            raise ValueError("Train and validation datasets must have the same class folders")
        if full_train.classes != test_dataset.classes:
            raise ValueError("Train and test datasets must have the same class folders")

        val_dataset = datasets.ImageFolder(str(val_path), transform=eval_transform)
        pin_memory = torch.cuda.is_available()
        return (
            DataLoader(full_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory),
            full_train.classes,
        )

    # Re-create the same images with deterministic transforms for validation.
    full_eval = datasets.ImageFolder(str(train_path), transform=eval_transform)

    n_total = len(full_train)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train,
        [n_train, n_val],
        generator=generator,
    )

    # Replace validation subset's dataset with deterministic-transform dataset.
    val_subset.dataset = full_eval

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, full_train.classes
