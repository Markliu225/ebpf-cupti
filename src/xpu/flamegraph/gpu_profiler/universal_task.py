import torch
import torch.cuda.nvtx as nvtx
import argparse
import sys
import time

# === 配置 ===
DEVICE = torch.device("cuda:0")

# 任务 A: 计算密集 (矩阵乘法)
def task_compute_A(a, b):
    torch.mm(a, b)

# 任务 B: 访存密集 (向量加法)
def task_memory_B(x, y):
    z = x + y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, choices=['A', 'B'], help="Task Type: A(Compute) or B(Memory)")
    parser.add_argument('--role', type=str, required=True, choices=['target', 'noise'], help="Role: target(measured) or noise(background)")
    parser.add_argument('--iters', type=int, default=50, help="Iterations for target")
    args = parser.parse_args()

    # 1. 数据初始化
    print(f"[{args.type}-{args.role}] Allocating Memory...")
    if args.type == 'A':
        # Matrix 4096 x 4096
        d1 = torch.randn(4096, 4096, device=DEVICE)
        d2 = torch.randn(4096, 4096, device=DEVICE)
        func = lambda: task_compute_A(d1, d2)
        nvtx_name = "Task_A_Compute"
    else:
        # Vector 50MB
        d1 = torch.randn(50_000_000, device=DEVICE)
        d2 = torch.randn(50_000_000, device=DEVICE)
        func = lambda: task_memory_B(d1, d2)
        nvtx_name = "Task_B_Memory"

    # 2. Warmup
    for _ in range(5):
        func()
    torch.cuda.synchronize()

    # 3. 执行逻辑
    if args.role == 'noise':
        print(f"[{args.type}-Noise] Starting Infinite Loop...")
        try:
            while True:
                func()
                # 稍微同步防止显存溢出，保持持续压力
                torch.cuda.synchronize()
        except KeyboardInterrupt:
            pass

    elif args.role == 'target':
        print(f"[{args.type}-Target] Starting Measurement ({args.iters} iters)...")
        
        # === NVTX 标记 (给 NCU 看) ===
        nvtx.range_push(nvtx_name)
        
        for i in range(args.iters):
            func()
            
        nvtx.range_pop()
        # ==========================
        
        print(f"[{args.type}-Target] Done.")

if __name__ == "__main__":
    main()