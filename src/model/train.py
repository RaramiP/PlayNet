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


def unfreeze_layers(model, unfreeze_from="layer3", verbose: bool = True):
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
        if verbose:
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
            if verbose:
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


def train_epoch(model, loader, criterion, optimizer, device: torch.device = DEVICE):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, threshold: float = 0.5, device: torch.device = DEVICE):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
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
# TRAINING WORKFLOW
# ============================================================


def run_training(
    csv_path: str = "data/dataset.csv",
    img_dir: str = "data/dataset_images",
    output_dir: str = ".",
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    num_epochs_frozen: int = NUM_EPOCHS_FROZEN,
    num_epochs_unfrozen: int = NUM_EPOCHS_UNFROZEN,
    device: torch.device = DEVICE,
    verbose: bool = True,
):
    """
    Run the full training workflow.

    Args:
        csv_path: Path to the dataset CSV file
        img_dir: Path to the directory containing images
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training
        num_workers: Number of data loader workers
        num_epochs_frozen: Epochs to train with frozen backbone
        num_epochs_unfrozen: Epochs to train with unfrozen layers
        device: Device to train on
        verbose: Whether to print progress

    Returns:
        dict with final metrics and model path
    """
    import os

    # Load data
    full_dataset = GameScreenshotDataset(
        csv_path=csv_path,
        img_dir=img_dir,
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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    if verbose:
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model with frozen backbone
    model = create_model(num_classes=len(FINAL_GENRES), freeze_backbone=True)
    model = model.to(device)

    trainable, total = count_parameters(model)
    if verbose:
        print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Loss with class weighting for imbalanced classes
    pos_weights = calculate_pos_weights(csv_path, FINAL_GENRES).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    phase1_path = os.path.join(output_dir, "best_model_phase1.pth")
    final_path = os.path.join(output_dir, "best_model_final.pth")

    # ========================================
    # PHASE 1: Train only classifier head
    # ========================================
    if verbose:
        print("\n" + "=" * 50)
        print("PHASE 1: Training classifier head (backbone frozen)")
        print("=" * 50)

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_f1 = 0

    for epoch in range(num_epochs_frozen):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device=device)

        scheduler.step(val_metrics["loss"])

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs_frozen}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(
                f"  Val Loss: {val_metrics['loss']:.4f}, Exact Match: {val_metrics['exact_match']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}"
            )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), phase1_path)

    # ========================================
    # PHASE 2: Unfreeze and fine-tune
    # ========================================
    if verbose:
        print("\n" + "=" * 50)
        print("PHASE 2: Fine-tuning (unfreezing layer3 and layer4)")
        print("=" * 50)

    # Load best model from phase 1
    model.load_state_dict(torch.load(phase1_path))

    # Unfreeze later layers
    unfreeze_layers(model, unfreeze_from="layer3", verbose=verbose)

    trainable, total = count_parameters(model)
    if verbose:
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

    for epoch in range(num_epochs_unfrozen):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device=device)

        scheduler.step(val_metrics["loss"])

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs_unfrozen}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(
                f"  Val Loss: {val_metrics['loss']:.4f}, Exact Match: {val_metrics['exact_match']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}"
            )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), final_path)
            if verbose:
                print("  -> New best model saved!")

    # ========================================
    # Final evaluation
    # ========================================
    if verbose:
        print("\n" + "=" * 50)
        print("FINAL EVALUATION")
        print("=" * 50)

    model.load_state_dict(torch.load(final_path))
    val_metrics = validate(model, val_loader, criterion, device=device)

    if verbose:
        print(f"Best Macro F1: {val_metrics['macro_f1']:.4f}")
        print("Per-label F1:")
        for genre, f1 in val_metrics["per_label_f1"].items():
            print(f"  {genre}: {f1:.4f}")

    return {
        "best_f1": best_f1,
        "final_metrics": val_metrics,
        "model_path": final_path,
    }
