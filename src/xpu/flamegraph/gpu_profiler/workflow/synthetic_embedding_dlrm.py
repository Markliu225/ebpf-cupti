import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def add_args(p) -> None:
    p.add_argument("--preset", choices=["small", "medium", "large"], default="small")
    p.add_argument("--num-tables", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--bag-size", type=int, default=8, help="indices per table per sample")
    p.add_argument("--mlp-dims", type=str, default="512,256", help="comma-separated MLP hidden dims")
    p.add_argument("--steps", type=int, default=50, help="steps per epoch when --limit-batches is not set")
    p.add_argument("--precision", choices=["float32", "amp"], default="amp")


def _preset(preset: str) -> Dict[str, int]:
    if preset == "medium":
        return {"num_tables": 32, "vocab_size": 50000}
    if preset == "large":
        return {"num_tables": 64, "vocab_size": 100000}
    return {"num_tables": 16, "vocab_size": 10000}


def _parse_dims(dims: str) -> List[int]:
    parts = [p.strip() for p in str(dims).split(",") if p.strip()]
    return [int(p) for p in parts] if parts else []


class DLRM(nn.Module):
    def __init__(self, num_tables: int, vocab_size: int, embed_dim: int, mlp_dims: List[int]):
        super().__init__()
        self.num_tables = num_tables
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.tables = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for _ in range(num_tables)])

        in_dim = num_tables * embed_dim
        dims = [in_dim] + mlp_dims + [1]
        layers: List[nn.Module] = []
        for d0, d1 in zip(dims, dims[1:]):
            layers.append(nn.Linear(d0, d1))
            if d1 != 1:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T, K)
        pooled: List[torch.Tensor] = []
        for t, emb in enumerate(self.tables):
            x = emb(idx[:, t, :])  # (B, K, D)
            pooled.append(x.mean(dim=1))
        cat = torch.cat(pooled, dim=1)
        return self.mlp(cat).squeeze(1)


def _num_steps(args) -> int:
    return int(args.limit_batches) if args.limit_batches else int(getattr(args, "steps", 50))


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.bfloat16


def _autocast_enabled(args, device: torch.device) -> bool:
    if getattr(args, "precision", "amp") != "amp":
        return False
    if device.type == "cpu":
        return False
    return True


def _build_batch(batch_size: int, num_tables: int, vocab_size: int, bag_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, vocab_size, (batch_size, num_tables, bag_size), device=device, dtype=torch.long)
    y = torch.randint(0, 2, (batch_size,), device=device, dtype=torch.float32)
    return idx, y


def train(args, device: torch.device, logger) -> None:
    cfg = _preset(args.preset)
    num_tables = int(args.num_tables or cfg["num_tables"])
    vocab_size = int(args.vocab_size or cfg["vocab_size"])
    embed_dim = int(args.embed_dim)
    bag_size = int(args.bag_size)
    mlp_dims = _parse_dims(args.mlp_dims)
    steps = _num_steps(args)

    model = DLRM(num_tables=num_tables, vocab_size=vocab_size, embed_dim=embed_dim, mlp_dims=mlp_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    use_autocast = _autocast_enabled(args, device)
    amp_dtype = _amp_dtype(device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            idx, y = _build_batch(args.batch_size, num_tables, vocab_size, bag_size, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                logits = model(idx)
                loss = criterion(logits.float(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        dt = time.perf_counter() - t0
        samples = steps * int(args.batch_size)
        logger.info(
            "train epoch=%d loss=%.6f steps=%d samples=%d samples_per_s=%.2f tables=%d vocab=%d dim=%d bag=%d amp=%s",
            epoch + 1,
            total_loss / max(1, steps),
            steps,
            samples,
            samples / max(1e-9, dt),
            num_tables,
            vocab_size,
            embed_dim,
            bag_size,
            use_autocast,
        )

    ckpt_path = Path(args.artifact_dir) / "synthetic_embedding_dlrm.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_tables": num_tables,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "bag_size": bag_size,
            "mlp_dims": mlp_dims,
        },
        ckpt_path,
    )
    logger.info("checkpoint saved path=%s", ckpt_path)


def infer(args, device: torch.device, logger) -> None:
    ckpt_path = args.checkpoint or Path(args.artifact_dir) / "synthetic_embedding_dlrm.pt"
    cfg = _preset(args.preset)
    num_tables = int(args.num_tables or cfg["num_tables"])
    vocab_size = int(args.vocab_size or cfg["vocab_size"])
    embed_dim = int(args.embed_dim)
    bag_size = int(args.bag_size)
    mlp_dims = _parse_dims(args.mlp_dims)

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        num_tables = int(checkpoint.get("num_tables", num_tables))
        vocab_size = int(checkpoint.get("vocab_size", vocab_size))
        embed_dim = int(checkpoint.get("embed_dim", embed_dim))
        bag_size = int(checkpoint.get("bag_size", bag_size))
        mlp_dims = checkpoint.get("mlp_dims", mlp_dims)

    steps = _num_steps(args)
    model = DLRM(num_tables=num_tables, vocab_size=vocab_size, embed_dim=embed_dim, mlp_dims=list(mlp_dims)).to(device)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    model.eval()

    use_autocast = _autocast_enabled(args, device)
    amp_dtype = _amp_dtype(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        acc = 0.0
        for _ in range(steps):
            idx, _ = _build_batch(args.batch_size, num_tables, vocab_size, bag_size, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                logits = model(idx)
            acc += logits.float().mean().item()
    dt = time.perf_counter() - t0
    samples = steps * int(args.batch_size)
    logger.info(
        "infer steps=%d samples=%d samples_per_s=%.2f mean_logit=%.6f tables=%d vocab=%d dim=%d bag=%d amp=%s",
        steps,
        samples,
        samples / max(1e-9, dt),
        acc / max(1, steps),
        num_tables,
        vocab_size,
        embed_dim,
        bag_size,
        use_autocast,
    )

