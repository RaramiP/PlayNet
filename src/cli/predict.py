import argparse

from src.model.predict import (
    DEVICE,
    load_model,
    predict,
    preprocess_image,
)


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
            print(f"  {r['genre']}: {r['probability']:.1%}")
    else:
        print("\nNo genres predicted above threshold.")

    print("\nAll probabilities:")
    for r in results:
        marker = "*" if r["predicted"] else " "
        print(f"  {marker} {r['genre']:25} {r['probability']:.1%}")


if __name__ == "__main__":
    main()
