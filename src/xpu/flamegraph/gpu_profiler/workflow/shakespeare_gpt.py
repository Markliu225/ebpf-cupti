import math
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def add_args(p) -> None:
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--prompt", type=str, default="To be, or not to be")
    p.add_argument("--gen-len", type=int, default=100)
    p.add_argument("--min-runtime-sec", type=float, default=5.0, help="ensure run lasts long enough for sampling")


class CharDataset(Dataset):
    def __init__(self, text: str, stoi: Dict[str, int], seq_len: int):
        self.data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class CharGRU(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        emb = self.embed(x)
        out, h = self.gru(emb, h)
        logits = self.fc(out)
        return logits, h


def download_dataset(data_root: Path) -> Path:
    target_dir = data_root / "shakespeare"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "tiny_shakespeare.txt"
    if not target_path.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, target_path)
    return target_path


def build_vocab(text: str) -> Tuple[Dict[str, int], List[str]]:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = chars
    return stoi, itos


def data_loader(data_root: Path, batch_size: int, seq_len: int, limit_batches: int) -> Tuple[DataLoader, Dict[str, int], List[str]]:
    path = download_dataset(data_root)
    text = path.read_text(encoding="utf-8")
    stoi, itos = build_vocab(text)
    dataset = CharDataset(text, stoi, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if limit_batches:
        loader = LimitedLoader(loader, limit_batches)
    return loader, stoi, itos


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
    loader, stoi, itos = data_loader(args.data_root, args.batch_size, args.seq_len, args.limit_batches or 0)
    vocab_size = len(stoi)
    model = CharGRU(vocab_size, args.hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    min_runtime = float(getattr(args, "min_runtime_sec", 0.0))

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(loader))
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        logger.info("train epoch=%s loss=%.4f ppl=%.2f", epoch + 1, avg_loss, ppl)
        while (time.perf_counter() - t0) < min_runtime:
            time.sleep(0.1)

    ckpt_path = Path(args.artifact_dir) / "shakespeare_gpt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "stoi": stoi, "itos": itos, "hidden_size": args.hidden_size}, ckpt_path)
    logger.info("checkpoint saved path=%s", ckpt_path)


def generate(model: nn.Module, stoi: Dict[str, int], itos: List[str], prompt: str, length: int, device: torch.device) -> str:
    model.eval()
    idxs = [stoi.get(ch, 0) for ch in prompt]
    input_ids = torch.tensor([idxs], dtype=torch.long, device=device)
    h = None
    generated = list(prompt)
    with torch.no_grad():
        for _ in range(length):
            logits, h = model(input_ids, h)
            next_logits = logits[:, -1, :]
            probs = torch.softmax(next_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            generated.append(itos[next_idx])
            input_ids = torch.tensor([[next_idx]], dtype=torch.long, device=device)
    return "".join(generated)


def infer(args, device: torch.device, logger) -> None:
    ckpt_path = args.checkpoint or Path(args.artifact_dir) / "shakespeare_gpt.pt"
    if not ckpt_path.exists():
        logger.error("checkpoint not found path=%s", ckpt_path)
        return

    min_runtime = float(getattr(args, "min_runtime_sec", 0.0))
    checkpoint = torch.load(ckpt_path, map_location=device)
    vocab_size = len(checkpoint["itos"])
    model = CharGRU(vocab_size, checkpoint["hidden_size"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    t0 = time.perf_counter()
    sample = generate(model, checkpoint["stoi"], checkpoint["itos"], args.prompt, args.gen_len, device)
    while (time.perf_counter() - t0) < min_runtime:
        time.sleep(0.1)
    logger.info("generated_text sample=%s", sample)
