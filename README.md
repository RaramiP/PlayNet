# PlayNet

Game genre classification from in-game screenshot images using deep learning.

PlayNet is a multi-label image classification system that predicts video game genres from screenshots. It leverages transfer learning with ResNet architectures trained on a curated dataset of Steam game screenshots spanning 12 genre categories.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
  - [Data Collection](#data-collection)
- [Model Architecture](#model-architecture)
  - [Transfer Learning Strategy](#transfer-learning-strategy)
  - [Data Augmentation](#data-augmentation)
  - [Class Imbalance Handling](#class-imbalance-handling)
- [Dataset](#dataset)
  - [Genre Taxonomy](#genre-taxonomy)
  - [Dataset Statistics](#dataset-statistics)
- [Pre-trained Models](#pre-trained-models)
- [License](#license)

## Overview

Given a screenshot of a video game, PlayNet outputs probability scores for each of 12 genre labels. Because a single game can belong to multiple genres simultaneously (for example, an action RPG), the system uses multi-label classification with independent sigmoid outputs rather than a single softmax.

The training pipeline implements a two-phase transfer learning approach: first training only a custom classifier head on frozen ResNet features, then selectively fine-tuning deeper layers with discriminative learning rates.

## Project Structure

```
PlayNet/
├── src/
│   ├── cli/
│   │   ├── train.py              # Training command-line interface
│   │   └── predict.py            # Inference command-line interface
│   ├── model/
│   │   ├── train.py              # Training pipeline and dataset loader
│   │   └── predict.py            # Inference pipeline and preprocessing
│   └── utils/
│       └── data_scrapping/
│           ├── steam_data_scrapping.py   # Steam API scraper
│           ├── download_image.py         # Screenshot downloader
│           └── make_dataset.py           # Dataset CSV generator
├── data/
│   ├── dataset.csv               # Training metadata and labels
│   └── dataset_images/           # Screenshot image files
├── notebooks/
│   └── resnet_test.ipynb         # Development and experimentation notebook
├── test_images/                  # Sample images for quick inference testing
├── pyproject.toml                # Project configuration and dependencies
├── requirements.txt              # Locked dependency versions
└── LICENSE                       # MIT License
```

## Requirements

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Hardware

- **Apple Silicon (recommended):** Training and inference leverage Metal Performance Shaders (MPS) for GPU acceleration.
- **CPU:** Supported as a fallback. Device selection is automatic.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/PlayNet.git
cd PlayNet
```

Using uv (recommended):

```bash
uv sync
```

Using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Inference

Run genre prediction on a single image:

```bash
python -m src.cli.predict path/to/screenshot.jpg --model best_resnet18.pth --threshold 0.5
```

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `image_path` | Yes | -- | Path to the input screenshot image |
| `--model` | No | `best_model_final.pth` | Path to a trained model checkpoint |
| `--threshold` | No | `0.5` | Classification threshold for genre predictions |

The output displays each genre with its predicted probability and whether it exceeds the threshold.

### Training

Train a new model from scratch using the dataset:

```bash
python -m src.cli.train \
  --csv-path data/dataset.csv \
  --img-dir data/dataset_images \
  --output-dir . \
  --batch-size 32 \
  --epochs-frozen 5 \
  --epochs-unfrozen 10
```

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--csv-path` | No | `data/dataset.csv` | Path to the dataset CSV file |
| `--img-dir` | No | `data/dataset_images` | Path to the image directory |
| `--output-dir` | No | `.` | Directory for saving model checkpoints |
| `--batch-size` | No | `32` | Training batch size |
| `--num-workers` | No | `4` | Number of data loading workers |
| `--epochs-frozen` | No | `5` | Number of epochs for Phase 1 (frozen backbone) |
| `--epochs-unfrozen` | No | `10` | Number of epochs for Phase 2 (fine-tuning) |
| `--quiet` | No | `false` | Suppress verbose training output |

Training produces two checkpoint files:

- `best_model_phase1.pth` -- Best model after Phase 1 (classifier head training)
- `best_model_final.pth` -- Best model after Phase 2 (fine-tuning)

### Data Collection

The data pipeline consists of three sequential steps. A Steam Web API key is required and should be placed in a `.env` file at the project root.

```env
API_KEY=your_steam_api_key_here
```

**Step 1: Scrape game metadata from Steam**

```bash
python src/utils/data_scrapping/steam_data_scrapping.py
```

Fetches game information (name, genres, screenshot URLs) from the Steam API. The scraper is resumable and checkpoints progress every 50 games.

**Step 2: Download screenshot images**

```bash
python src/utils/data_scrapping/download_image.py
```

Downloads up to 3 screenshots per game using 10 concurrent workers. Images are organized into per-game folders.

**Step 3: Generate the training dataset CSV**

```bash
python src/utils/data_scrapping/make_dataset.py
```

Maps raw Steam genres to the 12-class taxonomy and produces a CSV file with multi-hot encoded labels for each image.

## Model Architecture

PlayNet supports two backbone architectures:

| Architecture | Total Parameters | Checkpoint Size |
|---|---|---|
| ResNet18 | ~24.5M | ~44 MB |
| ResNet50 | ~25M | ~94 MB |

Both use pre-trained ImageNet weights as initialization. The final fully connected layer is replaced with a custom classifier head:

```
ResNet Backbone (pretrained on ImageNet)
    |
Linear(in_features -> 512) -> ReLU -> Dropout(0.3)
    |
Linear(512 -> 12) -> Sigmoid
```

The 12 output neurons correspond to independent genre probabilities.

### Transfer Learning Strategy

Training proceeds in two phases:

**Phase 1 -- Frozen Backbone**

All backbone layers are frozen. Only the classifier head is trained. This allows the new head to adapt to the feature space produced by the pre-trained backbone without disrupting learned representations.

- Optimizer: Adam (lr=1e-3)
- Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)
- Trainable parameters: ~1M

**Phase 2 -- Selective Fine-tuning**

The best Phase 1 checkpoint is loaded. The last two residual blocks (`layer3` and `layer4`) are unfrozen while earlier layers remain frozen. Discriminative learning rates are applied:

| Layer Group | Learning Rate |
|---|---|
| layer3 | 1e-5 |
| layer4 | 1e-5 |
| Classifier head | 1e-4 |

This preserves low-level feature detectors (edges, textures) from ImageNet while adapting higher-level features to game screenshot characteristics.

- Trainable parameters: ~5.7M

### Data Augmentation

Training images are augmented with the following transformations:

- Resize to 256x256, then random crop to 224x224
- Random horizontal flip (p=0.5)
- Color jitter (brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
- Random affine (rotation up to 10 degrees, translation up to 10%)
- Random grayscale (p=0.1)
- ImageNet normalization

Validation and inference use deterministic preprocessing: resize to 256x256, center crop to 224x224, and ImageNet normalization.

### Class Imbalance Handling

Genre distribution in the dataset is imbalanced (for example, Action and Indie games are far more common than Racing or Massively Multiplayer). To address this, positive class weights are computed from label frequencies and passed to `BCEWithLogitsLoss`. Weights are capped at 10.0 to prevent training instability.

## Dataset

### Genre Taxonomy

The system classifies screenshots into 12 genres:

| Genre | Description |
|---|---|
| Action | Fast-paced combat and gameplay |
| Free To Play | Free-to-play titles |
| Strategy | Turn-based or real-time strategy |
| Adventure | Story-driven exploration |
| Indie | Independent studio productions |
| RPG | Role-playing games |
| Casual | Casual gameplay experiences |
| Simulation | Simulation-based games |
| Racing | Vehicle and racing games |
| Massively Multiplayer | MMO titles |
| Sports | Sports-related games |
| Other | Genres outside the main 11 categories |

A single image may have multiple genre labels active simultaneously.

### Dataset Statistics

- **Total images:** 5,986
- **Unique games:** ~2,000
- **Screenshots per game:** Up to 3
- **Train/validation split:** 80% / 20%
- **Label format:** Multi-hot encoded binary vectors

## Pre-trained Models

Four checkpoints are included in the repository:

| File | Architecture | Phase | Description |
|---|---|---|---|
| `best_resnet18_phase1.pth` | ResNet18 | 1 | After classifier head training |
| `best_resnet18.pth` | ResNet18 | Final | After fine-tuning (recommended) |
| `best_resnet50_phase1.pth` | ResNet50 | 1 | After classifier head training |
| `best_resnet50.pth` | ResNet50 | Final | After fine-tuning |

The model architecture is automatically detected from the checkpoint filename during inference.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
