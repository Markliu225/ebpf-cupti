import torch
import torch.cuda.nvtx as nvtx

# 4096^2 矩阵乘法
N = 4096
a = torch.randn(N, N, device="cuda:0")
b = torch.randn(N, N, device="cuda:0")

# 预热
for _ in range(5): torch.mm(a, b)
torch.cuda.synchronize()

# 正式测量 (带标记)
nvtx.range_push("Region_A")
for _ in range(50):
    torch.mm(a, b)
nvtx.range_pop()

torch.cuda.synchronize()