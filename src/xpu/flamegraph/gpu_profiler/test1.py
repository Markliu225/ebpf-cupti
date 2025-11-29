#!/usr/bin/env python3
"""简单的 GPU 端深度学习示例，用于触发 CUDA kernel 以便 eBPF + CUPTI 采集。

特点：
- PyTorch MLP，使用随机合成数据做回归任务。
- 默认强制使用 CUDA 设备，以确保会产生足够的 kernel 调用。
- 每个 epoch 打印一次 loss、迭代耗时以及当前显存占用。

运行前请确认已经安装 GPU 版 PyTorch，例如：
  pip install torch --index-url https://download.pytorch.org/whl/cu121
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="简单的 GPU 端 MLP 训练示例")
    parser.add_argument("--batch-size", type=int, default=4096, help="批大小")
    parser.add_argument("--feature-dim", type=int, default=2048, help="输入特征维度")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="隐藏层维度")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--steps-per-epoch", type=int, default=50, help="每轮迭代步数")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument(
        "--allow-cpu", action="store_true", help="在没有 CUDA 时退回 CPU（默认报错）"
    )
    return parser.parse_args()


def build_model(input_dim: int, hidden_dim: int) -> nn.Module:
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    return model


def synthetic_batch(batch_size: int, feature_dim: int, device: torch.device):
    x = torch.randn(batch_size, feature_dim, device=device)
    # 随机生成一个线性映射作为“真值”，再加上噪声。
    w = torch.randn(feature_dim, 1, device=device)
    y = x @ w + 0.1 * torch.randn(batch_size, 1, device=device)
    return x, y


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("未检测到 CUDA，请在支持 GPU 的环境下运行，或添加 --allow-cpu")

    model = build_model(args.feature_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"Using device: {device} | params={sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0
        for step in range(args.steps_per_epoch):
            features, targets = synthetic_batch(args.batch_size, args.feature_dim, device)
            preds = model(features)
            loss = F.mse_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        torch.cuda.synchronize(device) if device.type == "cuda" else None
        elapsed = time.time() - t0
        avg_loss = epoch_loss / args.steps_per_epoch
        mem = (
            torch.cuda.memory_allocated(device) / 1024 ** 2
            if device.type == "cuda"
            else 0.0
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss={avg_loss:.5f} | time={elapsed:.3f}s | mem={mem:.1f} MiB"
        )

    print("训练结束，可搭配 cuda_cupti_integrator.py 同时运行以采集事件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
