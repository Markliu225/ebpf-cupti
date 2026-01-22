#!/bin/bash
set -euo pipefail

# === 配置 ===
PYTHON_BIN="/home/mark/anaconda3/envs/ebpf-cupti/bin/python"
NSYS_BIN="/usr/local/cuda/bin/nsys"
OUT_DIR="./results_nsys_power_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

echo "=== Starting Nsight Systems (Power Sequence) ==="
echo "Output Directory: $OUT_DIR"

# 检查 nsys
if [ ! -f "$NSYS_BIN" ]; then
    if command -v nsys &> /dev/null; then NSYS_BIN="nsys"; else echo "nsys not found"; exit 1; fi
fi

# 1. Profiling
echo ">>> Step 1: Profiling..."
sudo "$NSYS_BIN" profile \
    --trace=cuda,nvtx,osrt \
    --sample=process-tree \
    --gpu-metrics-device=all \
    --gpu-metrics-frequency=10000 \
    --output="$OUT_DIR/parallel_run" \
    --force-overwrite=true \
    "$PYTHON_BIN" "launcher.py"

# 2. Exporting to SQLite
echo ">>> Step 2: Exporting to SQLite..."

# 找到生成的 .nsys-rep 文件
REPORT_FILE=$(find "$OUT_DIR" -name "parallel_run*.nsys-rep" | head -n 1)

if [ -f "$REPORT_FILE" ]; then
    echo "Found report: $REPORT_FILE"
    
    # === [关键修改] ===
    # 1. 获取文件名 (去掉路径和后缀)，例如 "parallel_run"
    BASENAME=$(basename "$REPORT_FILE" .nsys-rep)
    
    # 2. 拼接目标路径，强制指定输出到 OUT_DIR
    OUTPUT_SQLITE="$OUT_DIR/$BASENAME.sqlite"
    
    echo "Exporting to: $OUTPUT_SQLITE"

    sudo "$NSYS_BIN" export \
        --type sqlite \
        --output "$OUTPUT_SQLITE" \
        --force-overwrite=true \
        "$REPORT_FILE"
        
    echo "Export successful."
else
    echo "Error: Report file not found."
    exit 1
fi

echo "=== Done ==="
echo "Files are located in: $OUT_DIR"