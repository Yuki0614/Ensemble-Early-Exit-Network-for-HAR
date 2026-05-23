# Ensemble Early-Exit Network for Human Activity Recognition

This repository contains the official implementation of **Ensemble Early-Exit Network for Human Activity Recognition Using Wearable Sensors**. The code implements an early-exit convolutional neural network for sensor-based human activity recognition (HAR), where intermediate classifiers can make predictions before the final layer when their confidence is high enough.

The project is designed for reproducible experiments on wearable-sensor HAR datasets such as UCI-HAR, PAMAP2, UniMiB SHAR, and OPPORTUNITY.

## Overview

Early-exit neural networks reduce inference cost by allowing easy samples to leave the network before all layers are evaluated. In this implementation:

- A five-layer CNN backbone is used for HAR time-series inputs.
- Intermediate classifiers are attached after Layer 2, Layer 3, and Layer 4.
- The final classifier is attached after Layer 5.
- During training, all exits are optimized jointly with a weighted loss.
- During testing, confidence thresholds decide whether a sample exits early.
- Ensemble prediction can combine current and previous exit predictions.

## Repository Structure

```text
.
|-- configs/
|   |-- config.yaml          # Global training/testing configuration
|   |-- uci.yaml             # UCI-HAR dataset configuration
|   |-- pamap2.yaml          # PAMAP2 dataset configuration template
|   |-- unimib.yaml          # UniMiB SHAR dataset configuration template
|   `-- oppo.yaml            # OPPORTUNITY dataset configuration template
|-- data/
|   `-- UCI/                 # Dataset files in .npy format
|-- outputs/
|   |-- checkpoints/         # Saved model checkpoints
|   |-- logs/                # Log files
|   `-- results/             # Evaluation metrics
|-- load_data.py             # Dataset loading and preprocessing
|-- model.py                 # CNN backbone, exits, and ensemble modules
|-- train.py                 # Training entry point
|-- test.py                  # Test-time early-exit evaluation
|-- utils.py                 # Metrics, checkpoint, seed, and helper functions
`-- requirements.txt
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- `numpy`
- `PyYAML`
- `torch`
- `thop`

## Data Preparation

Each dataset should be placed under `data/` with the following file names:

```text
data/UCI/
|-- train_x.npy
|-- train_y.npy
|-- test_x.npy
`-- test_y.npy
```

The expected input format is:

- `train_x.npy`, `test_x.npy`: `(samples, time_steps, sensor_channels)`
- `train_y.npy`, `test_y.npy`: `(samples,)` or one-hot labels

The data loader automatically reshapes 3D sensor inputs to:

```text
(samples, 1, time_steps, sensor_channels)
```

For UCI-HAR, if `data/UCI/UCI.zip` is available and the `.npy` files are missing, the loader will extract the archive automatically.

## Configuration

The global experiment settings are stored in `configs/config.yaml`:

```yaml
dataset: UCI

train:
  epochs: 300
  batch_size: 64
  test_batch_size: 1
  learning_rate: 0.0001
  weight_decay: 0.01
  seed: 77
  device: cuda
  num_workers: 0

loss:
  exit_weights: [1.0, 2.0, 3.0, 4.0]
```

Dataset-specific settings are stored in:

- `configs/uci.yaml`
- `configs/pamap2.yaml`
- `configs/unimib.yaml`
- `configs/oppo.yaml`

For example, the UCI-HAR configuration defines the input shape, number of classes, CNN layers, exit positions, and ensemble mode:

```yaml
input_shape: [128, 9]
num_classes: 6

model:
  in_channels: 1
  exit_layers: [2, 3, 4, 5]
  ensemble_mode: ensemble
```

`exit_layers: [2, 3, 4, 5]` means that classifiers are attached after Layer 2, Layer 3, Layer 4, and Layer 5. The first three are early exits, while Layer 5 is the final classifier.

The supported `ensemble_mode` values are:

- `none`: no ensemble; each exit predicts independently.
- `normal`: learnable affine transformation and summation of previous exit logits.
- `ensemble`: weighted geometric-mean ensemble using current and previous exit predictions.

## Training

Train the model with:

```bash
python train.py --config configs/config.yaml
```

To specify a dataset configuration manually:

```bash
python train.py --config configs/config.yaml --dataset-config configs/uci.yaml
```

The best checkpoint is saved to:

```text
outputs/checkpoints/UCI/net_EE_model.pt
```

Training and validation metrics are saved under:

```text
outputs/results/UCI/
```

## Testing

Evaluate a trained checkpoint with confidence-based early exit:

```bash
python test.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/UCI/net_EE_model.pt \
  --thresholds 0.8 0.8 0.8
```

The three threshold values correspond to the early exits after:

```text
Layer 2, Layer 3, Layer 4
```

Layer 5 is the final classifier and does not use a threshold. If a sample does not satisfy any early-exit threshold, it is classified by the final classifier.

To disable early exits and force all samples to use the final classifier:

```bash
python test.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/UCI/net_EE_model.pt \
  --thresholds 1.0 1.0 1.0
```

The test script reports accuracy, macro precision, macro recall, macro F1, and the exit distribution:

```text
Accuracy: 0.xxxx
Macro precision: 0.xxxx
Macro recall: 0.xxxx
Macro F1: 0.xxxx
Exit counts: {'exit_1': ..., 'exit_2': ..., 'exit_3': ..., 'final': ...}
```

Metrics are saved as JSON files in:

```text
outputs/results/UCI/
```

## Using Other Datasets

To run PAMAP2, UniMiB SHAR, or OPPORTUNITY:

1. Convert the dataset into `.npy` files named `train_x.npy`, `train_y.npy`, `test_x.npy`, and `test_y.npy`.
2. Place the files under the corresponding folder in `data/`.
3. Fill in `input_shape` and `num_classes` in the dataset configuration file.
4. Set `dataset` in `configs/config.yaml`.
5. Run `train.py` and `test.py`.

Example:

```yaml
dataset: PAMAP2
```

Then update:

```yaml
# configs/pamap2.yaml
input_shape: [time_steps, sensor_channels]
num_classes: number_of_classes
```

## Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@article{YU2025111409,
title = {Ensemble early exit network on human activity recognition using wearable sensors},
journal = {Computer Networks},
volume = {269},
pages = {111409},
year = {2025},
issn = {1389-1286},
doi = {https://doi.org/10.1016/j.comnet.2025.111409},
url = {https://www.sciencedirect.com/science/article/pii/S1389128625003767},
author = {Jianglai Yu and Lei Zhang and Dongzhou Cheng and Can Bu and Liangdong Liu and Hao Wu and Aiguo Song},
keywords = {Human activity recognition, Deep learning, Early-exit},
abstract = {Deep learning has recently achieved significant success in sensor-based human activity recognition (HAR). However, achieving higher levels of accuracy in these models often requires more computational resources. When such systems are deployed on edge devices with strict requirements for computing, memory, and communication, controlling energy consumption and inference latency becomes a challenging task. In the field of computer vision or natural language processing, it is recognized that not all inputs require the same amount of computation to produce reliable predictions. Adaptive inference, as a prominent approach for efficient deployment, has gained increasing attention. In particular, the ensemble early-exit mechanism has emerged as a promising direction for adjusting the computational depth of each input sample at runtime, providing better accuracy-cost trade-off for deep learning models. However, the “overthinking” problem has been rarely explored in the domain of HAR. In this paper, we for the first time leverage such ensemble early-exit mechanism to investigate the “overthinking” problem in the context of HAR. By cascading multiple exits in the early-exit network, we analyze the variations of activity prediction within the network, address the “overthinking” problem in early-exit activity recognition scenario, and propose strategies to mitigate it. Experimental results on multiple mainstream HAR benchmarks demonstrate that our approach allows for adjusting the computational load based on the difficulty levels of different sensor samples, leading to improved classification accuracy for various early-exit activity recognition tasks. A practical on-device latency analysis is provided. Our code will be released at: https://github.com/Yuki0614/Ensemble-Early-Exit-Network-for-HAR.}
}
```