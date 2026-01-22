#!/usr/bin/env python3
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx # [新增]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU MLP Task B")
    # [新增] 角色控制
    parser.add_argument("--role", type=str, default="target", choices=['target', 'noise'], 
                        help="运行角色: target(测量) 或 noise(背景干扰)")
    
    parser.add_argument("--batch-size", type=int, default=1, help="批大小")
    parser.add_argument("--feature-dim", type=int, default=256, help="输入特征维度")
    parser.add_argument("--hidden-dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--steps-per-epoch", type=int, default=50, help="每轮迭代步数")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--allow-cpu", action="store_true", help="允许CPU")
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
    w = torch.randn(feature_dim, 1, device=device)
    y = x @ w + 0.1 * torch.randn(batch_size, 1, device=device)
    return x, y

def train_epoch(model, optimizer, args, device, epoch_idx):
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
    if args.role == 'target' or (epoch_idx % 10 == 0):
        print(f"[Task B] Epoch {epoch_idx} | loss={avg_loss:.4f} | time={elapsed:.3f}s")

def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("No CUDA found.")

    model = build_model(args.feature_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[Task B] Role: {args.role} | Device: {device}")

    # === [核心逻辑修改] ===
    if args.role == 'noise':
        # 模式：噪声 (死循环)
        print("[Task B] Running infinite training loop as NOISE...")
        epoch_counter = 1
        try:
            while True:
                train_epoch(model, optimizer, args, device, epoch_counter)
                epoch_counter += 1
        except KeyboardInterrupt:
            pass
            
    else:
        # 模式：目标 (测量)
        print(f"[Task B] Running {args.epochs} epochs as TARGET...")
        
        # [关键] NVTX 开始
        nvtx.range_push("Region_B")
        
        for epoch in range(1, args.epochs + 1):
            train_epoch(model, optimizer, args, device, epoch)
            
        # [关键] NVTX 结束
        nvtx.range_pop()
        
    return 0

if __name__ == "__main__":
    raise SystemExit(main())