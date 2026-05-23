import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device(device_name="auto"):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def compute_classification_metrics(predictions, targets, num_classes):
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    accuracy = float((predictions == targets).mean()) if len(targets) else 0.0

    precision_list = []
    recall_list = []
    f1_list = []
    for class_index in range(num_classes):
        true_positive = np.logical_and(predictions == class_index, targets == class_index).sum()
        predicted_positive = (predictions == class_index).sum()
        target_positive = (targets == class_index).sum()

        precision = true_positive / (predicted_positive + 1e-12)
        recall = true_positive / (target_positive + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        precision_list.append(float(precision))
        recall_list.append(float(recall))
        f1_list.append(float(f1))

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_list)),
        "macro_recall": float(np.mean(recall_list)),
        "macro_f1": float(np.mean(f1_list)),
    }


def count_exit_distribution(exit_marks):
    counts = {
        "exit_1": 0,
        "exit_2": 0,
        "exit_3": 0,
        "final": 0,
    }
    for mark in exit_marks:
        if mark == 0:
            counts["final"] += 1
        elif mark in (1, 2, 3):
            counts[f"exit_{mark}"] += 1
        else:
            raise ValueError(f"Unexpected exit mark: {mark}")
    return counts


def save_checkpoint(model, path):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(model.state_dict(), path)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def count_para_flops(model, input_shape, device):
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError("Install thop to count FLOPs: pip install thop") from exc

    dummy_input = torch.randn(*input_shape).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    return {"flops": flops, "params": params}
