from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def add_args(p) -> None:
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-channels", type=int, default=128)
    p.add_argument("--min-runtime-sec", type=float, default=5.0, help="ensure run lasts long enough for sampling")


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int = 4, num_channels: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        kernel_sizes = (3, 4, 5)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_channels, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_classes)

    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        pooled = [torch.max(torch.relu(conv(embedded)), dim=2)[0] for conv in self.convs]
        cat = torch.cat(pooled, dim=1)
        out = self.dropout(cat)
        return self.fc(out)


def build_dataset(data_root: Path):
    try:
        from torchtext.datasets import AG_NEWS
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator
    except ImportError as exc:
        raise SystemExit("torchtext is required for agnews_textcnn; pip install torchtext") from exc

    tokenizer = get_tokenizer("basic_english")
    root = data_root / "ag_news"
    root.mkdir(parents=True, exist_ok=True)

    def yield_tokens(data_iter: Iterable[Tuple[int, str]]):
        for _, text in data_iter:
            yield tokenizer(text)

    try:
        # torchtext >=0.9
        train_iter = AG_NEWS(root=str(root), split="train")
        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        train_iter, test_iter = AG_NEWS(root=str(root), split=("train", "test"))
        train_list = list(train_iter)
        test_list = list(test_iter)
    except TypeError:
        # torchtext 0.6 legacy API (no split argument) – fall back to synthetic text samples
        vocab_size = 20000
        seq_len = 32
        def _make_split(n_samples: int) -> List[Tuple[int, torch.Tensor]]:
            data: List[Tuple[int, torch.Tensor]] = []
            for _ in range(n_samples):
                label = torch.randint(0, 4, (1,), dtype=torch.long).item() + 1  # labels 1..4 to mirror AG_NEWS
                tokens = torch.randint(1, vocab_size, (seq_len,), dtype=torch.long)
                data.append((label, tokens))
            return data

        train_list = _make_split(2000)
        test_list = _make_split(500)

        class _PreencodedVocab:
            def __init__(self, pad_idx: int, size: int):
                self.pad_idx = pad_idx
                self.size = size

            def __call__(self, tokens):
                if hasattr(tokens, "tolist"):
                    return list(tokens.tolist())
                return list(tokens)

            def __getitem__(self, item):
                if isinstance(item, str) and item == "<pad>":
                    return self.pad_idx
                if isinstance(item, str):
                    return self.pad_idx
                return int(item)

            def __len__(self):
                return self.size

        vocab = _PreencodedVocab(pad_idx=0, size=vocab_size)

    return tokenizer, vocab, train_list, test_list


def collate_batch(batch, tokenizer, vocab, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    labels: List[int] = []
    text_list: List[torch.Tensor] = []
    for label, text in batch:
        labels.append(label - 1)
        if isinstance(text, torch.Tensor):
            processed = text.long()
        else:
            processed = torch.tensor(vocab(tokenizer(text)), dtype=torch.long)
        text_list.append(processed)
    padded = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    return padded.to(device), torch.tensor(labels, dtype=torch.long, device=device)


def build_loaders(data_root: Path, batch_size: int, device: torch.device, limit_batches: int):
    tokenizer, vocab, train_list, test_list = build_dataset(data_root)
    collate = lambda batch: collate_batch(batch, tokenizer, vocab, device)
    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False, collate_fn=collate)

    if limit_batches:
        train_loader = LimitedLoader(train_loader, limit_batches)
        test_loader = LimitedLoader(test_loader, max(1, limit_batches // 2))
    return train_loader, test_loader, len(vocab)


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
    train_loader, test_loader, vocab_size = build_loaders(args.data_root, args.batch_size, device, args.limit_batches or 0)
    model = TextCNN(vocab_size=vocab_size, embed_dim=args.embed_dim, num_classes=4, num_channels=args.num_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    min_runtime = float(getattr(args, "min_runtime_sec", 0.0))

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.perf_counter()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_acc = correct / total if total else 0.0
        logger.info("train epoch=%s loss=%.4f acc=%.4f", epoch + 1, total_loss / max(1, len(train_loader)), train_acc)
        metrics = evaluate(model, test_loader, device)
        logger.info("eval epoch=%s loss=%.4f acc=%.4f", epoch + 1, metrics["loss"], metrics["acc"])
        while (time.perf_counter() - t0) < min_runtime:
            time.sleep(0.1)

    ckpt_path = Path(args.artifact_dir) / "agnews_textcnn.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "vocab_size": vocab_size, "embed_dim": args.embed_dim, "num_channels": args.num_channels}, ckpt_path)
    logger.info("checkpoint saved path=%s", ckpt_path)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_loss = total_loss / max(1, len(loader))
    acc = correct / total if total else 0.0
    return {"loss": avg_loss, "acc": acc}


def infer(args, device: torch.device, logger) -> None:
    ckpt_path = args.checkpoint or Path(args.artifact_dir) / "agnews_textcnn.pt"
    if not ckpt_path.exists():
        logger.error("checkpoint not found path=%s", ckpt_path)
        return

    min_runtime = float(getattr(args, "min_runtime_sec", 0.0))
    checkpoint = torch.load(ckpt_path, map_location=device)
    vocab_size = checkpoint["vocab_size"]
    model = TextCNN(vocab_size=vocab_size, embed_dim=checkpoint["embed_dim"], num_classes=4, num_channels=checkpoint.get("num_channels", 128)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    train_loader, test_loader, _ = build_loaders(args.data_root, args.batch_size, device, args.limit_batches or 0)
    t0 = time.perf_counter()
    metrics = evaluate(model, test_loader, device)
    while (time.perf_counter() - t0) < min_runtime:
        time.sleep(0.1)
    logger.info("inference metrics loss=%.4f acc=%.4f", metrics["loss"], metrics["acc"])
