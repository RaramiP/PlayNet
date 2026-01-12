import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

# ============================================================
# CONFIG
# ============================================================

FINAL_GENRES = [
    "Action",
    "Free To Play",
    "Strategy",
    "Adventure",
    "Indie",
    "RPG",
    "Casual",
    "Simulation",
    "Racing",
    "Massively Multiplayer",
    "Sports",
    "Other",
]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS_FROZEN = 5  # epochs with backbone frozen
NUM_EPOCHS_UNFROZEN = 10  # epochs with backbone unfrozen

# ============================================================
# DATASET
# ============================================================


class GameScreenshotDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")

        labels = torch.tensor(
            row[FINAL_GENRES].values.astype(float), dtype=torch.float32
        )

        if self.transform:
            image = self.transform(image)

        return image, labels


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ============================================================
# MODEL WITH FREEZING UTILITIES
# ============================================================


def create_model(num_classes, freeze_backbone=True):
    """Create ResNet18 with custom classifier head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all backbone layers
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier (always trainable)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


def unfreeze_layers(model, unfreeze_from="layer3"):
    """
    Gradually unfreeze layers.
    ResNet structure: conv1 -> bn1 -> layer1 -> layer2 -> layer3 -> layer4 -> fc

    unfreeze_from options:
        'layer4' - unfreeze only last residual block (conservative)
        'layer3' - unfreeze last two blocks (balanced)
        'layer2' - unfreeze more (aggressive)
        'all'    - unfreeze everything
    """
    if unfreeze_from == "all":
        for param in model.parameters():
            param.requires_grad = True
        print("Unfroze all layers")
        return

    # Define layer order
    layer_order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    unfreeze_idx = layer_order.index(unfreeze_from)
    layers_to_unfreeze = layer_order[unfreeze_idx:]

    for name, child in model.named_children():
        if name in layers_to_unfreeze:
            for param in child.parameters():
                param.requires_grad = True
            print(f"Unfroze {name}")

    # fc is always unfrozen
    for param in model.fc.parameters():
        param.requires_grad = True


def count_parameters(model):
    """Count trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def calculate_pos_weights(csv_path, genres):
    """Calculate positive weights for imbalanced classes."""
    df = pd.read_csv(csv_path)
    counts = df[genres].sum()
    total = len(df)

    # More weight for rare classes
    pos_weights = (total - counts) / (counts + 1)

    pos_weights = pos_weights.clip(upper=10)

    return torch.tensor(pos_weights.values, dtype=torch.float32)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = (torch.sigmoid(outputs) >= threshold).float()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Metrics
    exact_match = (all_preds == all_labels).all(dim=1).float().mean().item()

    # Per-label precision, recall, F1
    tp = (all_preds * all_labels).sum(dim=0).float()
    fp = (all_preds * (1 - all_labels)).sum(dim=0).float()
    fn = ((1 - all_preds) * all_labels).sum(dim=0).float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "loss": running_loss / len(loader.dataset),
        "exact_match": exact_match,
        "macro_f1": f1.mean().item(),
        "per_label_f1": dict(zip(FINAL_GENRES, f1.tolist())),
    }


# ============================================================
# MAIN TRAINING LOOP
# ============================================================


def main():
    # Load data
    full_dataset = GameScreenshotDataset(
        csv_path="data/dataset.csv",
        img_dir="data/dataset_images",
        transform=None,  # applied later
    )

    # Train/val split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model with frozen backbone
    model = create_model(num_classes=len(FINAL_GENRES), freeze_backbone=True)
    model = model.to(DEVICE)

    trainable, total = count_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Loss with class weighting for imbalanced classes
    pos_weights = calculate_pos_weights("data/dataset.csv", FINAL_GENRES).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # ========================================
    # PHASE 1: Train only classifier head
    # ========================================
    print("\n" + "=" * 50)
    print("PHASE 1: Training classifier head (backbone frozen)")
    print("=" * 50)

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_f1 = 0

    for epoch in range(NUM_EPOCHS_FROZEN):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_metrics = validate(model, val_loader, criterion)

        scheduler.step(val_metrics["loss"])

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS_FROZEN}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(
            f"  Val Loss: {val_metrics['loss']:.4f}, Exact Match: {val_metrics['exact_match']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), "best_model_phase1.pth")

    # ========================================
    # PHASE 2: Unfreeze and fine-tune
    # ========================================
    print("\n" + "=" * 50)
    print("PHASE 2: Fine-tuning (unfreezing layer3 and layer4)")
    print("=" * 50)

    # Load best model from phase 1
    model.load_state_dict(torch.load("best_model_phase1.pth"))

    # Unfreeze later layers
    unfreeze_layers(model, unfreeze_from="layer3")

    trainable, total = count_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Lower learning rate for fine-tuning
    optimizer = optim.Adam(
        [
            {"params": model.layer3.parameters(), "lr": 1e-5},
            {"params": model.layer4.parameters(), "lr": 1e-5},
            {"params": model.fc.parameters(), "lr": 1e-4},
        ]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    for epoch in range(NUM_EPOCHS_UNFROZEN):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_metrics = validate(model, val_loader, criterion)

        scheduler.step(val_metrics["loss"])

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS_UNFROZEN}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(
            f"  Val Loss: {val_metrics['loss']:.4f}, Exact Match: {val_metrics['exact_match']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), "best_model_final.pth")
            print("  -> New best model saved!")

    # ========================================
    # Final evaluation
    # ========================================
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)

    model.load_state_dict(torch.load("best_model_final.pth"))
    val_metrics = validate(model, val_loader, criterion)

    print(f"Best Macro F1: {val_metrics['macro_f1']:.4f}")
    print("Per-label F1:")
    for genre, f1 in val_metrics["per_label_f1"].items():
        print(f"  {genre}: {f1:.4f}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    main()
