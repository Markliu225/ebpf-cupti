import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def add_args(p) -> None:
    p.add_argument("--variant", choices=["tiny", "small", "base", "large"], default="tiny")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--steps", type=int, default=20, help="steps per epoch when --limit-batches is not set")
    p.add_argument("--amp", action="store_true", help="use autocast (fp16/bf16 depending on device)")
    p.add_argument("--channels-last", action="store_true", help="use channels_last memory format")


def _num_steps(args) -> int:
    return int(args.limit_batches) if args.limit_batches else int(getattr(args, "steps", 20))


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.bfloat16


def _build_model(variant: str, num_classes: int) -> nn.Module:
    fn = getattr(models, f"convnext_{variant}")
    return fn(weights=None, num_classes=num_classes)


def train(args, device: torch.device, logger) -> None:
    steps = _num_steps(args)
    variant = str(args.variant)
    num_classes = int(args.num_classes)
    model = _build_model(variant, num_classes).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    use_autocast = bool(args.amp) and device.type in {"cuda", "mps"}
    amp_dtype = _amp_dtype(device)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(use_autocast and device.type == "cuda"))

    bsz = int(args.batch_size)
    image_size = int(args.image_size)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            x = torch.randn(bsz, 3, image_size, image_size, device=device)
            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)
            y = torch.randint(0, num_classes, (bsz,), device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                logits = model(x)
                loss = criterion(logits, y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
        dt = time.perf_counter() - t0
        samples = steps * bsz
        logger.info(
            "train epoch=%d loss=%.6f steps=%d samples=%d samples_per_s=%.2f variant=%s amp=%s channels_last=%s",
            epoch + 1,
            total_loss / max(1, steps),
            steps,
            samples,
            samples / max(1e-9, dt),
            variant,
            use_autocast,
            bool(args.channels_last),
        )

    ckpt_path = Path(args.artifact_dir) / "convnext_benchmark.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "variant": variant, "num_classes": num_classes}, ckpt_path)
    logger.info("checkpoint saved path=%s", ckpt_path)


def infer(args, device: torch.device, logger) -> None:
    ckpt_path = args.checkpoint or Path(args.artifact_dir) / "convnext_benchmark.pt"
    variant = str(args.variant)
    num_classes = int(args.num_classes)
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        variant = checkpoint.get("variant", variant)
        num_classes = int(checkpoint.get("num_classes", num_classes))

    steps = _num_steps(args)
    model = _build_model(variant, num_classes).to(device)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    model.eval()

    use_autocast = bool(getattr(args, "amp", False)) and device.type in {"cuda", "mps"}
    amp_dtype = _amp_dtype(device)

    bsz = int(args.batch_size)
    image_size = int(args.image_size)
    t0 = time.perf_counter()
    with torch.no_grad():
        acc = 0.0
        for _ in range(steps):
            x = torch.randn(bsz, 3, image_size, image_size, device=device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                logits = model(x)
            acc += logits.float().mean().item()
    dt = time.perf_counter() - t0
    samples = steps * bsz
    logger.info(
        "infer steps=%d samples=%d samples_per_s=%.2f mean=%.6f variant=%s amp=%s",
        steps,
        samples,
        samples / max(1e-9, dt),
        acc / max(1, steps),
        variant,
        use_autocast,
    )

