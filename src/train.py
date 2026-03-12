from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

from src.config import IMAGE_SIZE, MODELS_DIR, RAW_GESTURE_DIR


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    if pretrained:
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(weights=None)
    else:
        model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / max(1, labels.size(0))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            total_acc += accuracy_from_logits(logits, labels)
            total_batches += 1

    return total_loss / max(1, total_batches), total_acc / max(1, total_batches)


def evaluate_classwise(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    num_classes = len(class_names)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            for true_label, pred_label in zip(labels, preds):
                confusion[int(true_label), int(pred_label)] += 1

    per_class_recall: list[float] = []
    for i in range(num_classes):
        total_true = int(confusion[i, :].sum().item())
        correct = int(confusion[i, i].item())
        recall = (correct / total_true) if total_true > 0 else 0.0
        per_class_recall.append(recall)

    return confusion, per_class_recall


def build_dataloaders(
    root: Path,
    image_size: tuple[int, int],
    val_split: float,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = datasets.ImageFolder(root=str(root), transform=train_tfms)
    if len(full_dataset) == 0:
        raise RuntimeError("No images found under data/raw/gestures")

    class_names = full_dataset.classes
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("Dataset split resulted in empty train/val set. Adjust --val-split")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Keep augmentations only for train split and deterministic preprocessing for val split.
    val_dataset = datasets.ImageFolder(root=str(root), transform=eval_tfms)
    val_subset.dataset = val_dataset

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_loader, val_loader, class_names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train gesture classifier")
    parser.add_argument("--data-dir", type=Path, default=RAW_GESTURE_DIR, help="Root class folder")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "gesture_classifier.pt")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    train_loader, val_loader, class_names = build_dataloaders(
        root=args.data_dir,
        image_size=IMAGE_SIZE,
        val_split=args.val_split,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Device: {device}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = run_epoch(
            model=model,
            loader=tqdm(train_loader, desc="train", leave=False),
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=tqdm(val_loader, desc="val", leave=False),
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        raise RuntimeError("Training ended without a valid checkpoint")

    checkpoint = {
        "model_name": "efficientnet_b0",
        "image_size": list(IMAGE_SIZE),
        "class_names": class_names,
        "state_dict": best_state,
        "best_val_acc": best_val_acc,
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
    }
    torch.save(checkpoint, args.out)

    model.load_state_dict(best_state)
    model.to(device)
    confusion, per_class_recall = evaluate_classwise(model, val_loader, class_names, device)

    classes_out = args.out.with_suffix(".classes.json")
    with classes_out.open("w", encoding="utf-8") as f:
        json.dump({"classes": class_names}, f, indent=2)

    print(f"\nSaved model: {args.out}")
    print(f"Saved class map: {classes_out}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("Validation recall by class:")
    for name, recall in zip(class_names, per_class_recall):
        print(f"- {name}: {recall:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion.tolist())


if __name__ == "__main__":
    main()
