import time
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def add_args(p) -> None:
    p.add_argument("--seq-len", type=int, default=72, help="sequence length (hours)")
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--min-runtime-sec", type=float, default=10.0, help="ensure run lasts long enough for sampling")


class SeriesDataset(Dataset):
    def __init__(self, series: torch.Tensor, seq_len: int):
        self.series = series
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.series) - self.seq_len)

    def __getitem__(self, idx):
        window = self.series[idx : idx + self.seq_len]
        target = self.series[idx + self.seq_len]
        return window.unsqueeze(-1), target.unsqueeze(-1)


class ForecastLSTM(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def download_jena(data_root: Path) -> Path:
    target_dir = data_root / "jena"
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "jena_climate_2009_2016.csv"
    if csv_path.exists():
        return csv_path

    zip_path = target_dir / "jena_climate_2009_2016.csv.zip"
    if not zip_path.exists():
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
        import urllib.request

        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    return csv_path


def load_series(data_root: Path) -> torch.Tensor:
    csv_path = download_jena(data_root)
    df = pd.read_csv(csv_path)
    temps = df["T (degC)"].values.astype("float32")
    temps = (temps - temps.mean()) / temps.std()
    return torch.tensor(temps, dtype=torch.float32)


def build_loaders(data_root: Path, batch_size: int, seq_len: int, limit_batches: int):
    series = load_series(data_root)
    split = int(len(series) * 0.8)
    train_ds = SeriesDataset(series[:split], seq_len)
    test_ds = SeriesDataset(series[split - seq_len :], seq_len)  # small overlap for context
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    if limit_batches:
        train_loader = LimitedLoader(train_loader, limit_batches)
        test_loader = LimitedLoader(test_loader, max(1, limit_batches // 2))
    return train_loader, test_loader


class LimitedLoader:
    def __init__(self, loader: DataLoader, limit: int):
        self.loader = loader
        self.limit = limit

    def __iter__(self):
        for idx, batch in enumerate(self.loader):
            if idx >= self.limit:
                break
            yield batch

    def __len__(self):
        return min(len(self.loader), self.limit)


def train(args, device: torch.device, logger) -> None:
    start_time = time.time()
    train_loader, test_loader = build_loaders(args.data_root, args.batch_size, args.seq_len, args.limit_batches or 0)
    model = ForecastLSTM(args.hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_rmse = (total_loss / max(1, len(train_loader))) ** 0.5
        metrics = evaluate(model, test_loader, device)
        logger.info("train epoch=%s rmse=%.4f eval_rmse=%.4f", epoch + 1, train_rmse, metrics["rmse"])

    ckpt_path = Path(args.artifact_dir) / "jena_lstm.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "hidden_size": args.hidden_size, "seq_len": args.seq_len}, ckpt_path)
    logger.info("checkpoint saved path=%s", ckpt_path)
    _ensure_min_runtime(start_time, args.min_runtime_sec, device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            total_loss += criterion(preds, targets).item()
            count += targets.numel()
    mse = total_loss / max(1, count)
    return {"rmse": mse ** 0.5}


def infer(args, device: torch.device, logger) -> None:
    start_time = time.time()
    ckpt_path = args.checkpoint or Path(args.artifact_dir) / "jena_lstm.pt"
    if not ckpt_path.exists():
        logger.error("checkpoint not found path=%s", ckpt_path)
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = ForecastLSTM(checkpoint.get("hidden_size", args.hidden_size)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    train_loader, test_loader = build_loaders(args.data_root, args.batch_size, checkpoint.get("seq_len", args.seq_len), args.limit_batches or 0)
    metrics = evaluate(model, test_loader, device)
    logger.info("inference rmse=%.4f", metrics["rmse"])
    _ensure_min_runtime(start_time, args.min_runtime_sec, device)


def _ensure_min_runtime(start_time: float, min_runtime_sec: float, device: torch.device) -> None:
    if min_runtime_sec <= 0:
        return
    warm_a = torch.randn(2048, 1024, device=device)
    warm_b = torch.randn(1024, 2048, device=device)
    while True:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.time() - start_time
        if elapsed >= min_runtime_sec:
            break
        torch.matmul(warm_a, warm_b)
