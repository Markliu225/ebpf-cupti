#!/bin/bash
set -euo pipefail

# === 配置路径 ===
# 务必确认这是你 Anaconda 环境的 Python
PYTHON_BIN="/home/mark/anaconda3/envs/ebpf-cupti/bin/python"
NCU_BIN="/usr/local/cuda/bin/ncu"

# 输出目录
OUT_DIR="./results_parallel_trace_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

# 关键指标 (同时抓取指令和显存)
METRICS="sm__inst_executed.sum,dram__bytes.sum,l2__misses.sum"

echo "=== 启动并行全量追踪 (Parallel Trace) ==="
echo "目标: 同时抓取 Task A 和 Task B 的数据序列"

# === 核心命令 ===
# --target-processes all: 这是关键！它告诉 ncu 只要是 launcher 启动的子崽子，都要抓！
# --app-replay-match grid: 确保并行回放时匹配正确 (可选，增加稳定性)
# python launcher.py: 让 launcher 去启动 A 和 B

sudo "$NCU_BIN" --target-processes all \
    --csv --log-file "$OUT_DIR/parallel_trace.csv" \
    --metrics "$METRICS" \
    --force-overwrite \
    "$PYTHON_BIN" "launcher.py"

echo "=== 完成 ==="
echo "请检查 $OUT_DIR/parallel_trace.csv"
echo "你应该能在 CSV 的 'Process ID' 列看到两个不同的 ID 交替出现。"