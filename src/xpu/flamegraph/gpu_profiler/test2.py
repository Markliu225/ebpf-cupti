import torch
import torch.cuda.nvtx as nvtx

# 200MB 向量加法
SIZE = 50_000_000
x = torch.randn(SIZE, device="cuda:0")
y = torch.randn(SIZE, device="cuda:0")

# 预热
for _ in range(5): z = x + y
torch.cuda.synchronize()

# 正式测量 (带标记)
nvtx.range_push("Region_B")
for _ in range(50):
    z = x + y
nvtx.range_pop()

torch.cuda.synchronize()