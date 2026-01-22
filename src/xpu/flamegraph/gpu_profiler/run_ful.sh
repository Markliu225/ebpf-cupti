#!/bin/bash
set -e  # 遇到错误立即停止

# ================= 配置区域 =================
# 1. 你的 Conda Python 路径 (请确认路径正确)
PYTHON_BIN="/home/mark/anaconda3/envs/ebpf-cupti/bin/python"

# 2. NCU 路径
NCU_BIN="/usr/local/cuda/bin/ncu"

# 3. 输出目录
OUT_DIR="./results_symmetric_metrics_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

# 4. === [关键更新] === 
# 你指定的 AccelWattch/GPGPU-Sim 风格详细指标列表
# 注意: ncu 要求逗号分隔，不能有空格
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

# ===========================================

echo "=== 开始对称并行实验 (详细指标版) ==="
echo "结果将保存至: $OUT_DIR"

# 检查工具是否存在
if [ ! -f "$NCU_BIN" ]; then
    echo "Error: NCU not found at $NCU_BIN"
    exit 1
fi

# -------------------------------------------------------
# 场景 1: 背景是 B (访存)，测量 A (计算)
# -------------------------------------------------------
echo ""
echo ">>> Phase 1: Running A (Target) under B (Noise)"

# 1.1 后台启动 B
echo "    [1/3] Starting Noise B..."
"$PYTHON_BIN" task_b_memory.py --role noise > /dev/null 2>&1 &
NOISE_PID=$!
sleep 4 # 等待 B 稳定

# 1.2 用 NCU 测量 A
echo "    [2/3] Profiling A..."
# 抓取 50 次 Kernel，形成序列
# 注意：这些 Realtime 指标非常适合测量并行环境下的真实吞吐
sudo "$NCU_BIN" \
    --csv --log-file "$OUT_DIR/A_under_B.csv" --force-overwrite \
    --target-processes all \
    --nvtx --nvtx-include "Task_A_Compute_Region/" \
    --launch-count 50 \
    --metrics "$METRICS" \
    "$PYTHON_BIN" task_a_compute.py --role target --iters 50

# 1.3 杀掉 B
echo "    [3/3] Stopping Noise B..."
kill $NOISE_PID
wait $NOISE_PID 2>/dev/null || true

# -------------------------------------------------------
# 场景 2: 背景是 A (计算)，测量 B (访存)
# -------------------------------------------------------
echo ""
echo ">>> Phase 2: Running B (Target) under A (Noise)"

# 2.1 后台启动 A
echo "    [1/3] Starting Noise A..."
"$PYTHON_BIN" task_a_compute.py --role noise > /dev/null 2>&1 &
NOISE_PID=$!
sleep 4 # 等待 A 稳定

# 2.2 用 NCU 测量 B
echo "    [2/3] Profiling B..."
sudo "$NCU_BIN" \
    --csv --log-file "$OUT_DIR/B_under_A.csv" --force-overwrite \
    --target-processes all \
    --nvtx --nvtx-include "Task_B_Memory_Region/" \
    --launch-count 50 \
    --metrics "$METRICS" \
    "$PYTHON_BIN" task_b_memory.py --role target --iters 50

# 2.3 杀掉 A
echo "    [3/3] Stopping Noise A..."
kill $NOISE_PID
wait $NOISE_PID 2>/dev/null || true

echo ""
echo "=== 完成 ==="
echo "请查看 $OUT_DIR 目录下的 CSV 文件。"
echo "注意: 如果出现 'Metric not found' 错误，请检查你的 GPU 架构是否支持该特定指标。"