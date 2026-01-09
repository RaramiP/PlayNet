import argparse
import multiprocessing

from src.model.train import (
    BATCH_SIZE,
    DEVICE,
    NUM_EPOCHS_FROZEN,
    NUM_EPOCHS_UNFROZEN,
    NUM_WORKERS,
    run_training,
)


def main():
    parser = argparse.ArgumentParser(description="Train game genre classification model")
    parser.add_argument(
        "--csv-path",
        default="data/dataset.csv",
        help="Path to dataset CSV (default: data/dataset.csv)",
    )
    parser.add_argument(
        "--img-dir",
        default="data/dataset_images",
        help="Path to images directory (default: data/dataset_images)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save model checkpoints (default: current directory)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of data loader workers (default: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--epochs-frozen",
        type=int,
        default=NUM_EPOCHS_FROZEN,
        help=f"Epochs with frozen backbone (default: {NUM_EPOCHS_FROZEN})",
    )
    parser.add_argument(
        "--epochs-unfrozen",
        type=int,
        default=NUM_EPOCHS_UNFROZEN,
        help=f"Epochs with unfrozen layers (default: {NUM_EPOCHS_UNFROZEN})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    result = run_training(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs_frozen=args.epochs_frozen,
        num_epochs_unfrozen=args.epochs_unfrozen,
        device=DEVICE,
        verbose=not args.quiet,
    )

    print(f"\nTraining complete! Best model saved to: {result['model_path']}")
    print(f"Best Macro F1: {result['best_f1']:.4f}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
