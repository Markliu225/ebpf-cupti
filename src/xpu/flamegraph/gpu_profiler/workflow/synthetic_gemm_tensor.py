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
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default=None, help="autocast dtype (default: device best)")


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


def _autocast_dtype(args, device: torch.device) -> torch.dtype:
    if args.amp_dtype:
        requested = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
        if device.type == "cpu" and requested == torch.float16:
            return torch.bfloat16
        if device.type == "mps" and requested == torch.bfloat16:
            return torch.float16
        return requested
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.bfloat16


def train(args, device: torch.device, logger) -> None:
    cfg = _preset(args.preset)
    hidden_size = int(args.hidden_size or cfg["hidden_size"])
    num_layers = int(args.num_layers or cfg["num_layers"])
    steps = _num_steps(args)

    amp_dtype = _autocast_dtype(args, device)
    use_autocast = device.type in {"cuda", "mps", "cpu"}
    use_scaler = device.type == "cuda"

    model = MLP(hidden_size=hidden_size, num_layers=num_layers).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_scaler)

    x = torch.randn(args.batch_size, hidden_size, device=device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                out = model(x)
                loss = out.float().pow(2).mean()
            if use_scaler:
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
            "train epoch=%d loss=%.6f steps=%d samples=%d samples_per_s=%.2f hidden=%d layers=%d amp_dtype=%s",
            epoch + 1,
            total_loss / max(1, steps),
            steps,
            samples,
            samples / max(1e-9, dt),
            hidden_size,
            num_layers,
            amp_dtype,
        )

    ckpt_path = Path(args.artifact_dir) / "synthetic_gemm_tensor.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "hidden_size": hidden_size, "num_layers": num_layers}, ckpt_path)
    logger.info("checkpoint saved path=%s", ckpt_path)


def infer(args, device: torch.device, logger) -> None:
    ckpt_path = args.checkpoint or Path(args.artifact_dir) / "synthetic_gemm_tensor.pt"
    cfg = _preset(args.preset)
    hidden_size = int(args.hidden_size or cfg["hidden_size"])
    num_layers = int(args.num_layers or cfg["num_layers"])
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        hidden_size = int(checkpoint.get("hidden_size", hidden_size))
        num_layers = int(checkpoint.get("num_layers", num_layers))

    steps = _num_steps(args)
    amp_dtype = _autocast_dtype(args, device)
    use_autocast = device.type in {"cuda", "mps", "cpu"}

    model = MLP(hidden_size=hidden_size, num_layers=num_layers).to(device=device)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    model.eval()

    x = torch.randn(args.batch_size, hidden_size, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        acc = 0.0
        for _ in range(steps):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                out = model(x)
            acc += out.float().mean().item()
    dt = time.perf_counter() - t0
    samples = steps * int(args.batch_size)
    logger.info(
        "infer steps=%d samples=%d samples_per_s=%.2f mean=%.6f hidden=%d layers=%d amp_dtype=%s",
        steps,
        samples,
        samples / max(1e-9, dt),
        acc / max(1, steps),
        hidden_size,
        num_layers,
        amp_dtype,
    )
