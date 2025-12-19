import zipfile
import time
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def add_args(p) -> None:
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--min-runtime-sec", type=float, default=5.0, help="ensure run lasts long enough for sampling")


class RatingsDataset(Dataset):
    def __init__(self, users: torch.Tensor, items: torch.Tensor, ratings: torch.Tensor):
        self.users = users
        self.items = items
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class MatrixFactorization(nn.Module):
    def __init__(self, num_users: int, num_items: int, embed_dim: int):
        super().__init__()
        self.user_factors = nn.Embedding(num_users, embed_dim)
        self.item_factors = nn.Embedding(num_items, embed_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, users, items):
        u = self.user_factors(users)
        v = self.item_factors(items)
        dot = (u * v).sum(dim=1)
        bias = self.user_bias(users).squeeze(1) + self.item_bias(items).squeeze(1) + self.global_bias
        return dot + bias


def download_movielens(data_root: Path) -> Path:
    target_dir = data_root / "movielens"
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted = target_dir / "ml-100k"
    if extracted.exists():
        return extracted

    zip_path = target_dir / "ml-100k.zip"
    if not zip_path.exists():
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        import urllib.request

        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    return extracted


def load_split(data_root: Path, test_ratio: float = 0.1) -> Tuple[Dataset, Dataset, Dict[str, int]]:
    data_dir = download_movielens(data_root)
    ratings_path = data_dir / "u.data"
    df = pd.read_csv(ratings_path, sep="\t", names=["user", "item", "rating", "timestamp"])
    user_ids = {uid: idx for idx, uid in enumerate(sorted(df.user.unique()))}
    item_ids = {iid: idx for idx, iid in enumerate(sorted(df.item.unique()))}

    users = df.user.map(user_ids).values
    items = df.item.map(item_ids).values
    ratings = df.rating.values.astype("float32")

    users_t = torch.tensor(users, dtype=torch.long)
    items_t = torch.tensor(items, dtype=torch.long)
    ratings_t = torch.tensor(ratings, dtype=torch.float32)

    split_idx = int(len(df) * (1 - test_ratio))
    train_ds = RatingsDataset(users_t[:split_idx], items_t[:split_idx], ratings_t[:split_idx])
    test_ds = RatingsDataset(users_t[split_idx:], items_t[split_idx:], ratings_t[split_idx:])
    meta = {"num_users": len(user_ids), "num_items": len(item_ids)}
    return train_ds, test_ds, meta


def build_loaders(data_root: Path, batch_size: int, device: torch.device, limit_batches: int):
    train_ds, test_ds, meta = load_split(data_root)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    if limit_batches:
        train_loader = LimitedLoader(train_loader, limit_batches)
        test_loader = LimitedLoader(test_loader, max(1, limit_batches // 2))
    return train_loader, test_loader, meta


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
    train_loader, test_loader, meta = build_loaders(args.data_root, args.batch_size, device, args.limit_batches or 0)
    model = MatrixFactorization(meta["num_users"], meta["num_items"], args.embed_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    min_runtime = float(getattr(args, "min_runtime_sec", 0.0))

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_rmse = (total_loss / max(1, len(train_loader))) ** 0.5
        metrics = evaluate(model, test_loader, device)
        logger.info("train epoch=%s rmse=%.4f eval_rmse=%.4f", epoch + 1, train_rmse, metrics["rmse"])
        t1.record()
        if min_runtime > 0:
            torch.cuda.synchronize()
            elapsed = t0.elapsed_time(t1) / 1000.0
            if elapsed < min_runtime:
                torch.cuda._sleep(int((min_runtime - elapsed) * 1e9))

    ckpt_path = Path(args.artifact_dir) / "movielens_mf.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "embed_dim": args.embed_dim, **meta}, ckpt_path)
    logger.info("checkpoint saved path=%s", ckpt_path)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for users, items, ratings in loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            preds = model(users, items)
            loss = criterion(preds, ratings)
            total_loss += loss.item()
            count += ratings.numel()
    mse = total_loss / max(1, count)
    return {"rmse": mse ** 0.5}


def infer(args, device: torch.device, logger) -> None:
    ckpt_path = args.checkpoint or Path(args.artifact_dir) / "movielens_mf.pt"
    if not ckpt_path.exists():
        logger.error("checkpoint not found path=%s", ckpt_path)
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    train_loader, test_loader, meta = build_loaders(args.data_root, args.batch_size, device, args.limit_batches or 0)
    model = MatrixFactorization(meta["num_users"], meta["num_items"], checkpoint.get("embed_dim", args.embed_dim)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    min_runtime = float(getattr(args, "min_runtime_sec", 0.0))
    start = time.perf_counter()
    metrics = evaluate(model, test_loader, device)
    # keep GPU busy to allow pm_sampling to grab samples
    while (time.perf_counter() - start) < min_runtime:
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)
        torch.matmul(a, b)
    logger.info("inference rmse=%.4f", metrics["rmse"])
