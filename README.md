# Horse Detection in Imagery

Object detection on a custom image dataset using Faster R-CNN and RetinaNet (PyTorch).

## Overview

This project explores object detection in thermal imagery for a single class (horse). The full pipeline was built from scratch: dataset collection and annotation, model fine-tuning, evaluation, and inference timing comparison between a two-stage and a one-stage detector.

## Dataset

- **200 images** collected and annotated manually using Label Studio
- Bounding box annotations exported in COCO format
- Single class: `horse`
- 160 / 40 train/test split (seed 42)

The dataset is not included in this repository due to size. To replicate, annotate your own images using [Label Studio](https://labelstud.io/) and export in COCO JSON format.

## Models

Both models were fine-tuned from ImageNet-pretrained weights with frozen backbones:

| Model | Architecture | Detector type |
|---|---|---|
| Faster R-CNN | ResNet50-FPN v2 | Two-stage |
| RetinaNet | ResNet50-FPN v2 | One-stage |

Training: 5 epochs, SGD (lr=0.005, momentum=0.9, weight_decay=0.0005)

## Results

| Model | mAP | mAP@50 | Mean inference (ms) |
|---|---|---|---|
| Faster R-CNN (two-stage) | **0.777** | **0.980** | 3568 |
| RetinaNet (one-stage) | 0.000 | 0.000 | 2043 |

**Faster R-CNN** achieved strong detection performance, with mAP@50 of 0.98 on the test set.

**RetinaNet** failed to converge on this dataset. This is likely due to the class imbalance handling in RetinaNet's focal loss being less effective on a small, single-class dataset with dense annotations — a known limitation of one-stage detectors in low-data regimes.

## Requirements

```
torch
torchvision
torchmetrics
pycocotools
faster-coco-eval
pandas
matplotlib
tqdm
Pillow
```

Install with:

```bash
pip install torch torchvision torchmetrics pycocotools faster-coco-eval pandas matplotlib tqdm Pillow
```

## Usage

Open `notebooks/horse_detection.ipynb` and update the following paths to point to your local dataset:

```python
COCO_JSON  = "path/to/result.json"
IMAGES_DIR = "path/to/images/"
```

Then run all cells in order.

## Tools & Libraries

- PyTorch 2.5.1
- TorchVision (Faster R-CNN, RetinaNet)
- TorchMetrics (mAP evaluation)
- PyCOCOTools (COCO format handling)
- Label Studio (dataset annotation)
