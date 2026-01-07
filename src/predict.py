import argparse

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


def create_model(num_classes, model_name: str = "resnet18"):
    """Create ResNet18 with custom classifier head."""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def load_model(checkpoint_path):
    """Load the trained model from checkpoint."""
    model = create_model(num_classes=len(FINAL_GENRES))
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image_path):
    """Preprocess image for inference."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


@torch.no_grad()
def predict(model, image_tensor, threshold=0.5):
    """Run inference and return predictions."""
    image_tensor = image_tensor.to(DEVICE)
    outputs = model(image_tensor)
    probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()

    results = []
    for genre, prob in zip(FINAL_GENRES, probabilities):
        results.append(
            {"genre": genre, "probability": float(prob), "predicted": prob >= threshold}
        )

    # Sort by probability descending
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict game genres from screenshot")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument(
        "--model",
        default="best_model_final.pth",
        help="Path to model checkpoint (default: best_model_final.pth)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold (default: 0.5)",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = load_model(args.model)

    print(f"Processing image: {args.image_path}")
    image_tensor = preprocess_image(args.image_path)

    print(f"Running inference on {DEVICE}...\n")
    results = predict(model, image_tensor, threshold=args.threshold)

    print("=" * 50)
    print("PREDICTIONS")
    print("=" * 50)

    predicted_genres = [r for r in results if r["predicted"]]
    if predicted_genres:
        print("\nPredicted genres:")
        for r in predicted_genres:
            print(f"  ✓ {r['genre']}: {r['probability']:.1%}")
    else:
        print("\nNo genres predicted above threshold.")

    print("\nAll probabilities:")
    for r in results:
        marker = "✓" if r["predicted"] else " "
        print(f"  {marker} {r['genre']:25} {r['probability']:.1%}")


if __name__ == "__main__":
    main()
