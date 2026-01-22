#!/bin/bash
set -e

# === 配置 ===
PYTHON_BIN="/home/mark/anaconda3/envs/ebpf-cupti/bin/python"
NCU_BIN="/usr/local/cuda/bin/ncu"
OUT_DIR="./results_symmetric"
mkdir -p "$OUT_DIR"

# 关键指标: 指令数(死数), 显存流量(会波动), L2 Miss(干扰的核心证据)
METRICS="sm__inst_executed.sum,dram__bytes.sum,l2__misses.sum"

echo "=== 开始对称干扰实验 ==="

# ==========================================
# 阶段 1: 背景是 B，测量 A
# ==========================================
echo ""
echo ">>> Phase 1: Noise = B, Target = A"

# 1.1 后台启动 B (Noise)
echo "    Starting Noise B..."
"$PYTHON_BIN" universal_task.py --type B --role noise > /dev/null 2>&1 &
NOISE_PID=$!
sleep 4 # 等待 B 跑满

# 1.2 用 NCU 测量 A (Target)
# 抓取 50 次连续的 A 的内核
echo "    Profiling A (under B's pressure)..."
sudo "$NCU_BIN" \
    --csv --log-file "$OUT_DIR/A_under_B.csv" --force-overwrite \
    --target-processes all \
    --nvtx --nvtx-include "Task_A_Compute/" \
    --launch-count 50 \
    --metrics "$METRICS" \
    "$PYTHON_BIN" universal_task.py --type A --role target --iters 50

# 1.3 杀掉 B
kill $NOISE_PID
wait $NOISE_PID 2>/dev/null || true


# ==========================================
# 阶段 2: 背景是 A，测量 B
# ==========================================
echo ""
echo ">>> Phase 2: Noise = A, Target = B"

# 2.1 后台启动 A (Noise)
echo "    Starting Noise A..."
"$PYTHON_BIN" universal_task.py --type A --role noise > /dev/null 2>&1 &
NOISE_PID=$!
sleep 4 # 等待 A 跑满

# 2.2 用 NCU 测量 B (Target)
echo "    Profiling B (under A's pressure)..."
sudo "$NCU_BIN" \
    --csv --log-file "$OUT_DIR/B_under_A.csv" --force-overwrite \
    --target-processes all \
    --nvtx --nvtx-include "Task_B_Memory/" \
    --launch-count 50 \
    --metrics "$METRICS" \
    "$PYTHON_BIN" universal_task.py --type B --role target --iters 50

# 2.3 杀掉 A
kill $NOISE_PID
wait $NOISE_PID 2>/dev/null || true

echo ""
echo "=== 实验结束 ==="
echo "数据已保存:"
echo "1. $OUT_DIR/A_under_B.csv (A 的指标序列)"
echo "2. $OUT_DIR/B_under_A.csv (B 的指标序列)"