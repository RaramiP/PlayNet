import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

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


def create_model(num_classes: int, model_name: str = "resnet18"):
    """Create ResNet18 with custom classifier head."""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def load_model(checkpoint_path: str, device: torch.device = DEVICE):
    """Load the trained model from checkpoint."""
    if "resnet18" in checkpoint_path:
        model_name = "resnet18"
    else:
        model_name = "resnet50"
    model = create_model(num_classes=len(FINAL_GENRES), model_name=model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def get_inference_transform():
    """Get the transform used for inference."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def preprocess_image(image_path: str):
    """Preprocess image for inference from file path."""
    transform = get_inference_transform()
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


def preprocess_pil_image(image: Image.Image):
    """Preprocess a PIL Image for inference."""
    transform = get_inference_transform()
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

@torch.no_grad()
def predict(model, image_tensor, threshold: float = 0.5, device: torch.device = DEVICE):
    """Run inference and return predictions."""
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()

    results = []
    for genre, prob in zip(FINAL_GENRES, probabilities):
        results.append(
            {
                "genre": str(genre),
                "probability": float(prob),                  # convert numpy float
                "predicted": bool(prob >= threshold),        # convert numpy bool -> Python bool
            }
        )

    # Sort by probability descending
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

