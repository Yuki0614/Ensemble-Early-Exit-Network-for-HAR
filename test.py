import argparse
from pathlib import Path

import torch

from load_data import build_dataloaders
from train import DATASET_CONFIGS, build_model, evaluate, print_metrics
from utils import get_device, load_yaml, save_json, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Test ensemble early-exit HAR model.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to global config file.")
    parser.add_argument("--dataset-config", default=None, help="Path to dataset config file.")
    parser.add_argument("--checkpoint", default=None, help="Path to trained model checkpoint (defaults to best checkpoint).")
    parser.add_argument(
        "--thresholds",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        metavar=("T2", "T3", "T4"),
        help="Confidence thresholds for exits after layer 2, layer 3, and layer 4.",
    )
    parser.add_argument("--output", default=None, help="Optional path to save test metrics JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    global_config = load_yaml(args.config)
    dataset_name = global_config["dataset"]
    dataset_config_path = args.dataset_config or DATASET_CONFIGS[dataset_name]
    dataset_config = load_yaml(dataset_config_path)

    if dataset_config["input_shape"] is None or dataset_config["num_classes"] is None:
        raise ValueError(f"Please fill input_shape and num_classes in {dataset_config_path}")

    train_config = global_config["train"]
    test_config = dict(train_config)
    test_config["test_batch_size"] = 1
    test_config["batch_size"] = train_config.get("batch_size", 64)

    setup_seed(train_config.get("seed", 77))
    device = get_device(train_config.get("device", "auto"))

    if args.checkpoint is None:
        checkpoint_dir = Path(global_config["output"]["checkpoint_dir"]) / dataset_name
        args.checkpoint = str(checkpoint_dir / "net_EE_model.pt")

    _, test_loader, _, test_dataset = build_dataloaders(dataset_config, test_config)
    model = build_model(
        dataset_config,
        global_config,
        exit_threshold=args.thresholds,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.set_exit_thresholds(args.thresholds)

    metrics = evaluate(model, test_loader, device, dataset_config["num_classes"])
    metrics["thresholds"] = {
        "layer2_exit": args.thresholds[0],
        "layer3_exit": args.thresholds[1],
        "layer4_exit": args.thresholds[2],
    }
    metrics["exit_policy"] = "confidence"
    metrics["test_samples"] = len(test_dataset)

    output_path = args.output
    if output_path is None:
        threshold_name = "_".join(str(threshold).replace(".", "p") for threshold in args.thresholds)
        output_path = Path(global_config["output"]["result_dir"]) / dataset_name / f"test_confidence_{threshold_name}.json"
    save_json(metrics, output_path)

    print(f"Dataset: {dataset_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Thresholds after layer 2/3/4: {args.thresholds}")
    print_metrics(metrics)
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
