from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from common import recommended_num_workers


def add_args(p) -> None:
    p.add_argument("--num-classes", type=int, default=10)


def build_loaders(data_root: Path, batch_size: int, device: torch.device, limit_batches: int) -> Tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.CIFAR10(root=data_root / "cifar10", train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_root / "cifar10", train=False, download=True, transform=test_tf)

    kwargs = {
        "batch_size": batch_size,
        "num_workers": recommended_num_workers(),
        "pin_memory": device.type == "cuda",
        "shuffle": True,
    }
    train_loader = DataLoader(train_set, **kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=kwargs["num_workers"], pin_memory=kwargs["pin_memory"], drop_last=False)

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


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(args, device: torch.device, logger) -> None:
    train_loader, test_loader = build_loaders(args.data_root, args.batch_size, device, args.limit_batches or 0)
    model = build_model(args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
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
        eval_metrics = evaluate(model, test_loader, device)
        logger.info("eval epoch=%s loss=%.4f acc=%.4f", epoch + 1, eval_metrics["loss"], eval_metrics["acc"])

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifact_dir / "cifar_resnet.pt"
    torch.save({"model_state": model.state_dict(), "num_classes": args.num_classes}, ckpt_path)
    logger.info("checkpoint saved path=%s", ckpt_path)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
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
    ckpt_path = args.checkpoint or Path(args.artifact_dir) / "cifar_resnet.pt"
    if not ckpt_path.exists():
        logger.error("checkpoint not found path=%s", ckpt_path)
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_model(checkpoint.get("num_classes", 10)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    _, test_loader = build_loaders(args.data_root, args.batch_size, device, args.limit_batches or 0)
    metrics = evaluate(model, test_loader, device)
    logger.info("inference metrics loss=%.4f acc=%.4f", metrics["loss"], metrics["acc"])

