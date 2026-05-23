from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class HARDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.data_x = self._load_x(self.x_path)
        self.data_y = self._load_y(self.y_path)

    @staticmethod
    def _load_x(path):
        data_x = np.load(path)
        if data_x.ndim == 3:
            data_x = data_x.reshape(-1, 1, data_x.shape[-2], data_x.shape[-1])
        elif data_x.ndim != 4:
            raise ValueError(f"Expected x data with 3 or 4 dims, got {data_x.shape}")
        return torch.from_numpy(data_x).float()

    @staticmethod
    def _load_y(path):
        data_y = np.load(path)
        if data_y.ndim > 1:
            data_y = data_y.argmax(axis=1)
        return torch.from_numpy(data_y).long()

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]


def prepare_dataset(dataset_config):
    data_dir = Path(dataset_config["data_dir"])
    expected_files = [
        data_dir / "train_x.npy",
        data_dir / "train_y.npy",
        data_dir / "test_x.npy",
        data_dir / "test_y.npy",
    ]
    if all(path.exists() for path in expected_files):
        return

    zip_path = Path(dataset_config.get("zip_path", data_dir / f"{dataset_config['name']}.zip"))
    if not zip_path.exists():
        missing = ", ".join(str(path) for path in expected_files if not path.exists())
        raise FileNotFoundError(f"Missing dataset files: {missing}")

    data_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(data_dir.parent)

    if not all(path.exists() for path in expected_files):
        missing = ", ".join(str(path) for path in expected_files if not path.exists())
        raise FileNotFoundError(f"Dataset archive was extracted, but files are still missing: {missing}")


def build_dataloaders(dataset_config, train_config):
    prepare_dataset(dataset_config)

    data_dir = Path(dataset_config["data_dir"])
    dataloader_config = dataset_config.get("dataloader", {})
    num_workers = train_config.get("num_workers", 0)

    train_dataset = HARDataset(data_dir / "train_x.npy", data_dir / "train_y.npy")
    test_dataset = HARDataset(data_dir / "test_x.npy", data_dir / "test_y.npy")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=dataloader_config.get("train_shuffle", True),
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=train_config.get("test_batch_size", 1),
        shuffle=dataloader_config.get("test_shuffle", False),
        num_workers=num_workers,
    )
    return train_loader, test_loader, train_dataset, test_dataset
