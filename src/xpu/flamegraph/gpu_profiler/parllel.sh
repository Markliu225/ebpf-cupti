#!/bin/bash
set -e

# === 用户配置区 ===
# 请修改为你的 Python 路径
PYTHON_BIN="/home/mark/anaconda3/envs/ebpf-cupti/bin/python"
NCU_BIN="/usr/local/cuda/bin/ncu"

# 你的 12 个指标
METRICS="dram__cycles_active.sum,\
dram__bytes.sum,\
sm__inst_executed_realtime.sum,\
l1tex__t_bytes.sum,\
sm__inst_executed_realtime.sum.per_cycle_active,\
sm__cycles_active.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_realtime.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_realtime.sum,\
l1tex__average_t_sector_pipe_lsu_mem_global_op_ld_lookup_miss.pct,\
lts__t_sectors_srcnode_gpc_op_read_realtime.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"

OUT_DIR="./results_mlp"
mkdir -p "$OUT_DIR"

echo "=== 开始 MLP 对称干扰实验 ==="

# -------------------------------
# 第一组: 背景跑 mlp_b (噪声)，测量 mlp_a (目标)
# -------------------------------
echo ">>> [1/2] 正在运行: 背景 MLP_B (噪声) + 前台 MLP_A (测量)"

# 1. 后台启动噪声
"$PYTHON_BIN" mlp_b.py --role noise > /dev/null 2>&1 &
NOISE_PID=$!
# MLP 启动比较慢（加载 PyTorch, 构建模型），多给点预热时间
sleep 5  

# 2. 前台测量目标
# 注意：--launch-count 20 意味着抓取前 20 个 Kernel。
# 对于 MLP，这可能只涵盖了第 1 个 epoch 的一部分。
sudo "$NCU_BIN" --csv --log-file "$OUT_DIR/A_under_B.csv" --force-overwrite \
    --target-processes all \
    --nvtx --nvtx-include "Region_A/" \
    --launch-count 20 \
    --metrics "$METRICS" \
    "$PYTHON_BIN" mlp_a.py --role target

# 3. 杀死噪声
kill $NOISE_PID && wait $NOISE_PID 2>/dev/null || true


# -------------------------------
# 第二组: 背景跑 mlp_a (噪声)，测量 mlp_b (目标)
# -------------------------------
echo ">>> [2/2] 正在运行: 背景 MLP_A (噪声) + 前台 MLP_B (测量)"

# 1. 后台启动噪声
"$PYTHON_BIN" mlp_a.py --role noise > /dev/null 2>&1 &
NOISE_PID=$!
sleep 5

# 2. 前台测量目标
sudo "$NCU_BIN" --csv --log-file "$OUT_DIR/B_under_A.csv" --force-overwrite \
    --target-processes all \
    --nvtx --nvtx-include "Region_B/" \
    --launch-count 20 \
    --metrics "$METRICS" \
    "$PYTHON_BIN" mlp_b.py --role target

# 3. 杀死噪声
kill $NOISE_PID && wait $NOISE_PID 2>/dev/null || true

echo "=== 完成 ==="
echo "结果位于: $OUT_DIR"