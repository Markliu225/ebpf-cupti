import torch
import torch.cuda.nvtx as nvtx
import argparse
import time
import sys

# === 配置: 访存密集型 (Vector Add) ===
# 5000万个 float 元素 ≈ 200MB 数据量
# 这远大于 L2 Cache (通常 6MB-80MB)，强制 GPU 频繁访问 DRAM
SIZE = 50_000_000 
DEVICE = torch.device("cuda:0")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', type=str, required=True, choices=['target', 'noise'], 
                        help="Role: 'target' for profiling, 'noise' for background interference")
    parser.add_argument('--iters', type=int, default=50, help="Number of iterations for target mode")
    args = parser.parse_args()

    # 1. 初始化数据
    print(f"[Task B] Allocating memory for VectorAdd ({SIZE} elements)...")
    try:
        x = torch.randn(SIZE, device=DEVICE)
        y = torch.randn(SIZE, device=DEVICE)
    except Exception as e:
        print(f"[Error] GPU Memory Init failed: {e}")
        sys.exit(1)

    # 2. 预热
    for _ in range(5):
        z = x + y
    torch.cuda.synchronize()

    # === 模式 1: 背景噪声 (Noise) ===
    if args.role == 'noise':
        print("[Task B-Noise] Starting Infinite Loop (Press Ctrl+C to stop)...")
        try:
            while True:
                z = x + y
                torch.cuda.synchronize()
        except KeyboardInterrupt:
            print("\n[Task B-Noise] Stopped.")

    # === 模式 2: 测量目标 (Target) ===
    elif args.role == 'target':
        print(f"[Task B-Target] Starting Measurement ({args.iters} iters)...")
        
        # [关键] NVTX 标记
        nvtx.range_push("Task_B_Memory_Region")
        
        for i in range(args.iters):
            z = x + y
            
        nvtx.range_pop()
        
        torch.cuda.synchronize()
        print("[Task B-Target] Done.")

if __name__ == "__main__":
    main()