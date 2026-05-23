import argparse
from pathlib import Path

import torch
from torch import nn

from load_data import build_dataloaders
from model import Net_EE
from utils import (
    compute_classification_metrics,
    ensure_dir,
    get_device,
    load_yaml,
    save_checkpoint,
    save_json,
    setup_seed,
    count_exit_distribution,
)


DATASET_CONFIGS = {
    "UCI": "configs/uci.yaml",
    "PAMAP2": "configs/pamap2.yaml",
    "UniMiB": "configs/unimib.yaml",
    "OPPO": "configs/oppo.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train ensemble early-exit HAR model.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to global config file.")
    parser.add_argument("--dataset-config", default=None, help="Path to dataset config file.")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for evaluation or resume.")
    return parser.parse_args()


def build_model(dataset_config, global_config=None, exit_threshold=None):
    model_config = dataset_config.get("model", {})
    exit_threshold = exit_threshold or [1.0, 1.0, 1.0]
    return Net_EE(
        train_shape=dataset_config["input_shape"],
        category=dataset_config["num_classes"],
        exit_threshold=exit_threshold,
        in_channels=model_config.get("in_channels", 1),
        layers=model_config.get("layers"),
        backbone_channels=model_config.get("backbone_channels", [16, 64, 64, 128]),
        exit_layers=model_config.get("exit_layers", [2, 3, 4, 5]),
        ensemble_mode=model_config.get("ensemble_mode", "normal"),
    )


def compute_loss(outputs, targets, criterion, exit_weights):
    if len(exit_weights) != len(outputs):
        raise ValueError(f"exit_weights length ({len(exit_weights)}) must match outputs length ({len(outputs)})")
    total_loss = 0.0
    total_weight = 0.0
    for output, weight in zip(outputs, exit_weights):
        weight = float(weight)
        total_loss = total_loss + criterion(output, targets) * weight
        total_weight += weight
    return total_loss / total_weight


def train_one_epoch(model, train_loader, criterion, optimizer, device, exit_weights):
    model.train()

    running_loss = 0.0
    correct_by_exit = [0, 0, 0, 0]
    total_samples = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).long()

        outputs = model(batch_x)
        loss = compute_loss(outputs, batch_y, criterion, exit_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch_y.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        for index, output in enumerate(outputs):
            correct_by_exit[index] += (output.argmax(1) == batch_y).sum().item()

    return {
        "loss": running_loss / max(total_samples, 1),
        "exit_acc": [correct / max(total_samples, 1) for correct in correct_by_exit],
    }


def evaluate(model, test_loader, device, num_classes):
    model.eval()

    exit_counts = [0, 0, 0, 0]
    exit_correct = [0, 0, 0, 0]
    exit_marks = []
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).long()

            outputs, exit_mark = model(batch_x)
            logits = outputs[-1]
            pred = logits.argmax(1)

            exit_counts[exit_mark] += batch_y.size(0)
            exit_correct[exit_mark] += (pred == batch_y).sum().item()
            exit_marks.extend([exit_mark] * batch_y.size(0))
            predictions.extend(pred.cpu().numpy().tolist())
            targets.extend(batch_y.cpu().numpy().tolist())

    metrics = compute_classification_metrics(predictions, targets, num_classes)
    metrics["exit_counts"] = count_exit_distribution(exit_marks)
    metrics["exit_accuracy"] = {
        "final": exit_correct[0] / max(exit_counts[0], 1),
        "exit_1": exit_correct[1] / max(exit_counts[1], 1),
        "exit_2": exit_correct[2] / max(exit_counts[2], 1),
        "exit_3": exit_correct[3] / max(exit_counts[3], 1),
    }
    return metrics


def main():
    args = parse_args()
    global_config = load_yaml(args.config)
    dataset_name = global_config["dataset"]
    dataset_config_path = args.dataset_config or DATASET_CONFIGS[dataset_name]
    dataset_config = load_yaml(dataset_config_path)

    if dataset_config["input_shape"] is None or dataset_config["num_classes"] is None:
        raise ValueError(f"Please fill input_shape and num_classes in {dataset_config_path}")

    train_config = global_config["train"]
    output_config = global_config["output"]
    setup_seed(train_config.get("seed", 77))
    device = get_device(train_config.get("device", "auto"))

    train_loader, test_loader, train_dataset, test_dataset = build_dataloaders(dataset_config, train_config)
    model = build_model(dataset_config, global_config).to(device)

    checkpoint_dir = Path(output_config["checkpoint_dir"]) / dataset_name
    result_dir = Path(output_config["result_dir"]) / dataset_name
    ensure_dir(checkpoint_dir)
    ensure_dir(result_dir)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.eval_only:
        metrics = evaluate(model, test_loader, device, dataset_config["num_classes"])
        save_json(metrics, result_dir / "eval_metrics.json")
        print_metrics(metrics)
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    exit_weights = global_config.get("loss", {}).get("exit_weights", [1.0, 2.0, 3.0, 4.0])

    best_accuracy = 0.0
    print(f"Dataset: {dataset_name}")
    print(f"Train samples: {len(train_dataset)}, test samples: {len(test_dataset)}")
    print(f"Device: {device}")

    for epoch in range(1, train_config["epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, exit_weights)
        eval_metrics = evaluate(model, test_loader, device, dataset_config["num_classes"])

        print(
            f"Epoch [{epoch}/{train_config['epochs']}] "
            f"loss={train_metrics['loss']:.4f} "
            f"acc={eval_metrics['accuracy']:.4f} "
            f"macro_f1={eval_metrics['macro_f1']:.4f}"
        )

        if eval_metrics["accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["accuracy"]
            save_checkpoint(model, checkpoint_dir / "net_EE_model.pt")
            save_json(eval_metrics, result_dir / "best_metrics.json")
            print("Saved best checkpoint.")

    final_metrics = evaluate(model, test_loader, device, dataset_config["num_classes"])
    save_json(final_metrics, result_dir / "final_metrics.json")
    print_metrics(final_metrics)


def print_metrics(metrics):
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Exit counts: {metrics['exit_counts']}")
    print(f"Exit accuracy: {metrics['exit_accuracy']}")


if __name__ == "__main__":
    main()
