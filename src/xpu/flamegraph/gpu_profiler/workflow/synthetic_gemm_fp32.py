import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim


def add_args(p) -> None:
    p.add_argument("--preset", choices=["small", "medium", "large"], default="small")
    p.add_argument("--hidden-size", type=int, default=None, help="override hidden size (d_model)")
    p.add_argument("--num-layers", type=int, default=None, help="override number of linear layers")
    p.add_argument("--steps", type=int, default=50, help="steps per epoch when --limit-batches is not set")
    p.add_argument("--allow-tf32", action="store_true", help="allow TF32 on CUDA (default: disabled)")


def _preset(preset: str) -> Dict[str, int]:
    if preset == "medium":
        return {"hidden_size": 2048, "num_layers": 8}
    if preset == "large":
        return {"hidden_size": 4096, "num_layers": 12}
    return {"hidden_size": 1024, "num_layers": 4}


class MLP(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _num_steps(args) -> int:
    return int(args.limit_batches) if args.limit_batches else int(getattr(args, "steps", 50))


def train(args, device: torch.device, logger) -> None:
    cfg = _preset(args.preset)
    hidden_size = int(args.hidden_size or cfg["hidden_size"])
    num_layers = int(args.num_layers or cfg["num_layers"])
    steps = _num_steps(args)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    model = MLP(hidden_size=hidden_size, num_layers=num_layers).to(device=device, dtype=torch.float32)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    x = torch.randn(args.batch_size, hidden_size, device=device, dtype=torch.float32)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = out.float().pow(2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        dt = time.perf_counter() - t0
        samples = steps * int(args.batch_size)
        logger.info(
            "train epoch=%d loss=%.6f steps=%d samples=%d samples_per_s=%.2f hidden=%d layers=%d",
            epoch + 1,
            total_loss / max(1, steps),
            steps,
            samples,
            samples / max(1e-9, dt),
            hidden_size,
            num_layers,
        )

    ckpt_path = Path(args.artifact_dir) / "synthetic_gemm_fp32.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "hidden_size": hidden_size, "num_layers": num_layers}, ckpt_path)
    logger.info("checkpoint saved path=%s", ckpt_path)


def infer(args, device: torch.device, logger) -> None:
    ckpt_path = args.checkpoint or Path(args.artifact_dir) / "synthetic_gemm_fp32.pt"
    cfg = _preset(args.preset)
    hidden_size = int(args.hidden_size or cfg["hidden_size"])
    num_layers = int(args.num_layers or cfg["num_layers"])
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        hidden_size = int(checkpoint.get("hidden_size", hidden_size))
        num_layers = int(checkpoint.get("num_layers", num_layers))

    steps = _num_steps(args)
    model = MLP(hidden_size=hidden_size, num_layers=num_layers).to(device=device, dtype=torch.float32)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    model.eval()

    x = torch.randn(args.batch_size, hidden_size, device=device, dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        acc = 0.0
        for _ in range(steps):
            out = model(x)
            acc += out.float().mean().item()
    dt = time.perf_counter() - t0
    samples = steps * int(args.batch_size)
    logger.info(
        "infer steps=%d samples=%d samples_per_s=%.2f mean=%.6f hidden=%d layers=%d",
        steps,
        samples,
        samples / max(1e-9, dt),
        acc / max(1, steps),
        hidden_size,
        num_layers,
    )

