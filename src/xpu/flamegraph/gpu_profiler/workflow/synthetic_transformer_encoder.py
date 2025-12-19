import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim


def add_args(p) -> None:
    p.add_argument("--preset", choices=["small", "medium", "large"], default="small")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--nhead", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--ffn-dim", type=int, default=None)
    p.add_argument("--steps", type=int, default=50, help="steps per epoch when --limit-batches is not set")
    p.add_argument("--amp", action="store_true", help="use autocast (fp16/bf16 depending on device)")


def _preset(preset: str) -> Dict[str, int]:
    if preset == "medium":
        return {"seq_len": 512, "d_model": 512, "nhead": 8, "num_layers": 8, "ffn_dim": 2048}
    if preset == "large":
        return {"seq_len": 1024, "d_model": 768, "nhead": 12, "num_layers": 12, "ffn_dim": 3072}
    return {"seq_len": 256, "d_model": 384, "nhead": 6, "num_layers": 6, "ffn_dim": 1536}


class EncoderClassifier(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, nhead: int, num_layers: int, ffn_dim: int, num_classes: int):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.encoder(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


def _num_steps(args) -> int:
    return int(args.limit_batches) if args.limit_batches else int(getattr(args, "steps", 50))


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.bfloat16


def _resolve_hparams(args, checkpoint_hparams: Dict[str, object] | None = None) -> Dict[str, object]:
    checkpoint_hparams = checkpoint_hparams or {}
    preset = str(checkpoint_hparams.get("preset", args.preset))
    cfg = _preset(preset)
    return {
        "preset": preset,
        "vocab_size": int(checkpoint_hparams.get("vocab_size", getattr(args, "vocab_size", 32000))),
        "num_classes": int(checkpoint_hparams.get("num_classes", getattr(args, "num_classes", 1000))),
        "seq_len": int(checkpoint_hparams.get("seq_len", int(args.seq_len or cfg["seq_len"]))),
        "d_model": int(checkpoint_hparams.get("d_model", int(args.d_model or cfg["d_model"]))),
        "nhead": int(checkpoint_hparams.get("nhead", int(args.nhead or cfg["nhead"]))),
        "num_layers": int(checkpoint_hparams.get("num_layers", int(args.num_layers or cfg["num_layers"]))),
        "ffn_dim": int(checkpoint_hparams.get("ffn_dim", int(args.ffn_dim or cfg["ffn_dim"]))),
    }


def _build_model(hparams: Dict[str, object]) -> EncoderClassifier:
    return EncoderClassifier(
        vocab_size=int(hparams["vocab_size"]),
        seq_len=int(hparams["seq_len"]),
        d_model=int(hparams["d_model"]),
        nhead=int(hparams["nhead"]),
        num_layers=int(hparams["num_layers"]),
        ffn_dim=int(hparams["ffn_dim"]),
        num_classes=int(hparams["num_classes"]),
    )


def train(args, device: torch.device, logger) -> None:
    steps = _num_steps(args)
    hparams = _resolve_hparams(args)
    model = _build_model(hparams).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    use_autocast = bool(args.amp) and device.type in {"cuda", "mps"}
    amp_dtype = _amp_dtype(device)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(use_autocast and device.type == "cuda"))

    seq_len = int(hparams["seq_len"])
    vocab_size = int(hparams["vocab_size"])
    num_classes = int(hparams["num_classes"])

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            input_ids = torch.randint(0, vocab_size, (int(args.batch_size), seq_len), device=device, dtype=torch.long)
            labels = torch.randint(0, num_classes, (int(args.batch_size),), device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                logits = model(input_ids)
                loss = criterion(logits, labels)
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
            "train epoch=%d loss=%.6f steps=%d samples=%d samples_per_s=%.2f seq_len=%d d_model=%d layers=%d amp=%s",
            epoch + 1,
            total_loss / max(1, steps),
            steps,
            samples,
            samples / max(1e-9, dt),
            seq_len,
            int(hparams["d_model"]),
            int(hparams["num_layers"]),
            use_autocast,
        )

    ckpt_path = Path(args.artifact_dir) / "synthetic_transformer_encoder.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "hyperparams": hparams}, ckpt_path)
    logger.info("checkpoint saved path=%s", ckpt_path)


def infer(args, device: torch.device, logger) -> None:
    ckpt_path = args.checkpoint or Path(args.artifact_dir) / "synthetic_transformer_encoder.pt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        hparams = _resolve_hparams(args, checkpoint.get("hyperparams") or {})
    else:
        checkpoint = None
        hparams = _resolve_hparams(args)

    steps = _num_steps(args)
    model = _build_model(hparams).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
    model.eval()

    seq_len = int(hparams["seq_len"])
    vocab_size = int(hparams["vocab_size"])

    use_autocast = bool(getattr(args, "amp", False)) and device.type in {"cuda", "mps"}
    amp_dtype = _amp_dtype(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        acc = 0.0
        for _ in range(steps):
            input_ids = torch.randint(0, vocab_size, (int(args.batch_size), seq_len), device=device, dtype=torch.long)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                logits = model(input_ids)
            acc += logits.float().mean().item()
    dt = time.perf_counter() - t0
    samples = steps * int(args.batch_size)
    logger.info(
        "infer steps=%d samples=%d samples_per_s=%.2f mean=%.6f seq_len=%d d_model=%d layers=%d amp=%s",
        steps,
        samples,
        samples / max(1e-9, dt),
        acc / max(1, steps),
        seq_len,
        int(hparams["d_model"]),
        int(hparams["num_layers"]),
        use_autocast,
    )
