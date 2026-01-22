import torch
import torch.cuda.nvtx as nvtx
import time
import argparse
import sys

# 配置：向量加法
SIZE = 50_000_000 
device = torch.device("cuda:0")

def run_task(mode, iters):
    x = torch.randn(SIZE, device=device)
    y = torch.randn(SIZE, device=device)
    print(f"[Task B] Initialized Vector Add ({SIZE} elements)")

    # Warmup
    for _ in range(5):
        z = x + y
    torch.cuda.synchronize()

    # === 模式 1: 充当背景噪声 ===
    if mode == 'noise':
        print("[Task B] Running as BACKGROUND NOISE (Infinite Loop)...")
        try:
            while True:
                z = x + y
                torch.cuda.synchronize()
        except KeyboardInterrupt:
            print("[Task B] Background loop stopped.")

    # === 模式 2: 充当测量目标 ===
    elif mode == 'measure':
        print(f"[Task B] Running as TARGET ({iters} iters)...")
        
        nvtx.range_push("Task_B_Region")
        
        start = time.time()
        
        for _ in range(iters):
            z = x + y
        
        torch.cuda.synchronize()
        end = time.time()
        
        nvtx.range_pop()
        
        avg_time = (end - start) / iters
        print(f"RESULT_B_TIME: {avg_time:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['measure', 'noise'])
    parser.add_argument('--iters', type=int, default=50)
    args = parser.parse_args()
    
    run_task(args.mode, args.iters)