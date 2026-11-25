"""Train a ResNet-18 image classifier on your own folder-based dataset.

Usage:
    python train.py

Expects images in:
    dataset/train/<class_name>/*.jpg
    dataset/val/<class_name>/*.jpg

Saves the best model (weights + class names) to custom_cnn.pth.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

DATA_DIR = Path("dataset")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
CHECKPOINT_PATH = Path("custom_cnn.pth")

NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = min(4, os.cpu_count() or 1)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def get_device() -> torch.device:
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders() -> tuple[DataLoader, DataLoader, list[str]]:
    """Load dataset/train and dataset/val with ImageFolder and wrap them in DataLoaders."""
    for split_dir in (TRAIN_DIR, VAL_DIR):
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing dataset folder: {split_dir.resolve()}\n"
                "Create dataset/train/<class_name>/ and dataset/val/<class_name>/ "
                "and put images inside."
            )

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=TRAIN_TRANSFORM)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=VAL_TRANSFORM)

    if len(train_dataset) == 0:
        raise RuntimeError(
            f"No training images found in {TRAIN_DIR.resolve()}.\n"
            "Add image files inside the class subfolders, e.g. "
            "dataset/train/class_a/photo1.jpg"
        )
    if len(val_dataset) == 0:
        raise RuntimeError(
            f"No validation images found in {VAL_DIR.resolve()}.\n"
            "Add image files inside the class subfolders, e.g. "
            "dataset/val/class_a/photo1.jpg"
        )

    class_names = train_dataset.classes
    if val_dataset.classes != class_names:
        raise RuntimeError(
            "Class folders differ between train and val.\n"
            f"  train: {class_names}\n"
            f"  val:   {val_dataset.classes}\n"
            "Both splits must contain exactly the same folder names."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return train_loader, val_loader, class_names


def build_model(num_classes: int) -> nn.Module:
    """Load a pre-trained ResNet-18, freeze its base, and swap the classifier head."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one pass over `loader`. Trains when an optimizer is given, else evaluates."""
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return running_loss / total, running_correct / total


def main() -> None:
    torch.manual_seed(42)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = build_dataloaders()
    print(f"Found {len(train_loader.dataset)} training images and "
          f"{len(val_loader.dataset)} validation images in {len(class_names)} classes: "
          f"{class_names}")

    model = build_model(len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                },
                CHECKPOINT_PATH,
            )
            print(f"  -> saved new best model (val acc {val_acc:.4f}) to {CHECKPOINT_PATH}")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint written to {CHECKPOINT_PATH.resolve()}")


if __name__ == "__main__":
    main()
