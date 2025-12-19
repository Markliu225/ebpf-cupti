import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def add_args(p) -> None:
    p.add_argument("--preset", choices=["small", "medium", "large"], default="small")
    p.add_argument("--num-categorical", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--num-dense", type=int, default=13)
    p.add_argument("--cross-layers", type=int, default=3)
    p.add_argument("--mlp-dims", type=str, default="256,128", help="comma-separated MLP hidden dims")
    p.add_argument("--steps", type=int, default=50, help="steps per epoch when --limit-batches is not set")
    p.add_argument("--amp", action="store_true", help="use autocast (fp16/bf16 depending on device)")
    p.add_argument("--min-runtime-sec", type=float, default=10.0, help="ensure run lasts long enough for sampling")


def _preset(preset: str) -> Dict[str, int]:
    if preset == "medium":
        return {"num_categorical": 26, "vocab_size": 50000}
    if preset == "large":
        return {"num_categorical": 39, "vocab_size": 100000}
    return {"num_categorical": 16, "vocab_size": 20000}


def _parse_dims(dims: str) -> List[int]:
    parts = [p.strip() for p in str(dims).split(",") if p.strip()]
    return [int(p) for p in parts] if parts else []


class CrossNet(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim)) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for w, b in zip(self.weights, self.biases):
            xw = (x * w).sum(dim=1, keepdim=True)
            x = x0 * xw + b + x
        return x


class DCNv2CTR(nn.Module):
    def __init__(
        self,
        num_categorical: int,
        vocab_size: int,
        embed_dim: int,
        num_dense: int,
        cross_layers: int,
        mlp_dims: List[int],
    ):
        super().__init__()
        self.num_categorical = num_categorical
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_dense = num_dense

        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for _ in range(num_categorical)])
        input_dim = num_dense + num_categorical * embed_dim
        self.cross = CrossNet(input_dim=input_dim, num_layers=cross_layers)

        deep_dims = [input_dim] + mlp_dims
        deep_layers: List[nn.Module] = []
        for d0, d1 in zip(deep_dims, deep_dims[1:]):
            deep_layers.append(nn.Linear(d0, d1))
            deep_layers.append(nn.ReLU())
        self.deep = nn.Sequential(*deep_layers) if deep_layers else nn.Identity()

        deep_out_dim = deep_dims[-1] if mlp_dims else input_dim
        self.out = nn.Linear(input_dim + deep_out_dim, 1)

    def forward(self, dense: torch.Tensor, cats: torch.Tensor) -> torch.Tensor:
        # dense: (B, Fd), cats: (B, Fc)
        embs: List[torch.Tensor] = []
        for i, emb in enumerate(self.embeddings):
            embs.append(emb(cats[:, i]))
        cat = torch.cat(embs, dim=1) if embs else torch.empty((dense.size(0), 0), device=dense.device)
        x0 = torch.cat([dense, cat], dim=1)
        x_cross = self.cross(x0)
        x_deep = self.deep(x0)
        x = torch.cat([x_cross, x_deep], dim=1)
        return self.out(x).squeeze(1)


def _num_steps(args) -> int:
    return int(args.limit_batches) if args.limit_batches else int(getattr(args, "steps", 50))


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.bfloat16


def _build_batch(batch_size: int, num_dense: int, num_categorical: int, vocab_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dense = torch.randn(batch_size, num_dense, device=device, dtype=torch.float32)
    cats = torch.randint(0, vocab_size, (batch_size, num_categorical), device=device, dtype=torch.long)
    y = torch.randint(0, 2, (batch_size,), device=device, dtype=torch.float32)
    return dense, cats, y


def train(args, device: torch.device, logger) -> None:
    start_time = time.time()
    cfg = _preset(args.preset)
    num_categorical = int(args.num_categorical or cfg["num_categorical"])
    vocab_size = int(args.vocab_size or cfg["vocab_size"])
    embed_dim = int(args.embed_dim)
    num_dense = int(args.num_dense)
    cross_layers = int(args.cross_layers)
    mlp_dims = _parse_dims(args.mlp_dims)
    steps = _num_steps(args)

    model = DCNv2CTR(
        num_categorical=num_categorical,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_dense=num_dense,
        cross_layers=cross_layers,
        mlp_dims=mlp_dims,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    use_autocast = bool(args.amp) and device.type in {"cuda", "mps"}
    amp_dtype = _amp_dtype(device)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(use_autocast and device.type == "cuda"))

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            dense, cats, y = _build_batch(int(args.batch_size), num_dense, num_categorical, vocab_size, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                logits = model(dense, cats)
                loss = criterion(logits.float(), y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
        dt = time.perf_counter() - t0
        samples = steps * int(args.batch_size)
        logger.info(
            "train epoch=%d loss=%.6f steps=%d samples=%d samples_per_s=%.2f cats=%d vocab=%d embed=%d cross=%d amp=%s",
            epoch + 1,
            total_loss / max(1, steps),
            steps,
            samples,
            samples / max(1e-9, dt),
            num_categorical,
            vocab_size,
            embed_dim,
            cross_layers,
            use_autocast,
        )

    ckpt_path = Path(args.artifact_dir) / "dcnv2_ctr.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_categorical": num_categorical,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_dense": num_dense,
            "cross_layers": cross_layers,
            "mlp_dims": mlp_dims,
        },
        ckpt_path,
    )
    logger.info("checkpoint saved path=%s", ckpt_path)
    _ensure_min_runtime(start_time, getattr(args, "min_runtime_sec", 0.0), device)


def infer(args, device: torch.device, logger) -> None:
    start_time = time.time()
    ckpt_path = args.checkpoint or Path(args.artifact_dir) / "dcnv2_ctr.pt"
    cfg = _preset(args.preset)
    num_categorical = int(args.num_categorical or cfg["num_categorical"])
    vocab_size = int(args.vocab_size or cfg["vocab_size"])
    embed_dim = int(args.embed_dim)
    num_dense = int(args.num_dense)
    cross_layers = int(args.cross_layers)
    mlp_dims = _parse_dims(args.mlp_dims)

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        num_categorical = int(checkpoint.get("num_categorical", num_categorical))
        vocab_size = int(checkpoint.get("vocab_size", vocab_size))
        embed_dim = int(checkpoint.get("embed_dim", embed_dim))
        num_dense = int(checkpoint.get("num_dense", num_dense))
        cross_layers = int(checkpoint.get("cross_layers", cross_layers))
        mlp_dims = checkpoint.get("mlp_dims", mlp_dims)
    else:
        checkpoint = None

    steps = _num_steps(args)
    model = DCNv2CTR(
        num_categorical=num_categorical,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_dense=num_dense,
        cross_layers=cross_layers,
        mlp_dims=list(mlp_dims),
    ).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
    model.eval()

    use_autocast = bool(getattr(args, "amp", False)) and device.type in {"cuda", "mps"}
    amp_dtype = _amp_dtype(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        acc = 0.0
        for _ in range(steps):
            dense, cats, _ = _build_batch(int(args.batch_size), num_dense, num_categorical, vocab_size, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                logits = model(dense, cats)
            acc += logits.float().mean().item()
    dt = time.perf_counter() - t0
    samples = steps * int(args.batch_size)
    logger.info(
        "infer steps=%d samples=%d samples_per_s=%.2f mean_logit=%.6f cats=%d vocab=%d embed=%d cross=%d amp=%s",
        steps,
        samples,
        samples / max(1e-9, dt),
        acc / max(1, steps),
        num_categorical,
        vocab_size,
        embed_dim,
        cross_layers,
        use_autocast,
    )
    _ensure_min_runtime(start_time, getattr(args, "min_runtime_sec", 0.0), device)


def _ensure_min_runtime(start_time: float, min_runtime_sec: float, device: torch.device) -> None:
    if min_runtime_sec <= 0:
        return
    a = torch.randn(2048, 1024, device=device)
    b = torch.randn(1024, 2048, device=device)
    while True:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if time.time() - start_time >= min_runtime_sec:
            break
        torch.matmul(a, b)
