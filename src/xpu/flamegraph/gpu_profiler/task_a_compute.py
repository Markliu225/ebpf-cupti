import torch
import torch.cuda.nvtx as nvtx
import argparse
import time
import sys

# === 配置: 计算密集型 (Matrix Multiplication) ===
# 4096 x 4096 的矩阵乘法足以让 SM 满载
N = 4096 
DEVICE = torch.device("cuda:0")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', type=str, required=True, choices=['target', 'noise'], 
                        help="Role: 'target' for profiling, 'noise' for background interference")
    parser.add_argument('--iters', type=int, default=50, help="Number of iterations for target mode")
    args = parser.parse_args()

    # 1. 初始化数据
    print(f"[Task A] Allocating memory for GEMM ({N}x{N})...")
    try:
        a = torch.randn(N, N, device=DEVICE)
        b = torch.randn(N, N, device=DEVICE)
    except Exception as e:
        print(f"[Error] GPU Memory Init failed: {e}")
        sys.exit(1)

    # 2. 预热 (Warmup) - 激活 GPU 频率
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize()

    # === 模式 1: 背景噪声 (Noise) ===
    if args.role == 'noise':
        print("[Task A-Noise] Starting Infinite Loop (Press Ctrl+C to stop)...")
        try:
            while True:
                torch.mm(a, b)
                # 适当同步，防止指令队列堆积过深导致 OOM，保持稳态压力
                torch.cuda.synchronize()
        except KeyboardInterrupt:
            print("\n[Task A-Noise] Stopped.")

    # === 模式 2: 测量目标 (Target) ===
    elif args.role == 'target':
        print(f"[Task A-Target] Starting Measurement ({args.iters} iters)...")
        
        # [关键] NVTX 标记开始：告诉 ncu 从这里开始抓数据
        nvtx.range_push("Task_A_Compute_Region")
        
        for i in range(args.iters):
            torch.mm(a, b)
            
        # [关键] NVTX 标记结束
        nvtx.range_pop()
        
        # 确保所有 Kernel 执行完毕
        torch.cuda.synchronize()
        print("[Task A-Target] Done.")

if __name__ == "__main__":
    main()