import torch
import torch.cuda.nvtx as nvtx
import time
import argparse
import sys

# 配置：矩阵乘法
N = 4096 
device = torch.device("cuda:0")

def run_task(mode, iters):
    # 初始化
    a = torch.randn(N, N, device=device)
    b = torch.randn(N, N, device=device)
    print(f"[Task A] Initialized GEMM ({N}x{N})")

    # Warmup
    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()

    # === 模式 1: 充当背景噪声 (Noise) ===
    if mode == 'noise':
        print("[Task A] Running as BACKGROUND NOISE (Infinite Loop)...")
        try:
            while True:
                torch.mm(a, b)
                # 适当同步防止 Command Buffer 溢出
                torch.cuda.synchronize() 
        except KeyboardInterrupt:
            print("[Task A] Background loop stopped.")

    # === 模式 2: 充当测量目标 (Target) ===
    elif mode == 'measure':
        print(f"[Task A] Running as TARGET ({iters} iters)...")
        
        # NVTX 标记范围 (给 ncu 看)
        nvtx.range_push("Task_A_Region")
        
        # 计时 (给 T_real 看)
        start = time.time()
        
        for _ in range(iters):
            torch.mm(a, b)
        
        torch.cuda.synchronize()
        end = time.time()
        
        nvtx.range_pop()
        
        avg_time = (end - start) / iters
        print(f"RESULT_A_TIME: {avg_time:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['measure', 'noise'])
    parser.add_argument('--iters', type=int, default=50)
    args = parser.parse_args()
    
    run_task(args.mode, args.iters)