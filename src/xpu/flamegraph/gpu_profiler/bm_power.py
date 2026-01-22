import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import csv

# === 模型系数配置 (根据你的显卡微调) ===
MODEL_CONFIG = {
    "P_IDLE": 0,          # 静态闲置功耗 (W)
    "W_SM": 4e-6,          # SM 指令权值
    "W_MEM": 1.0e-4,         # 显存字节权值
    "W_CACHE": 1.0e-4,       # 缓存权值
    "W_ACT": 1.0e-4          # 活跃周期权值
}

def parse_ncu_csv(filepath):
    """
    鲁棒的 NCU CSV 解析器：自动跳过 ==PROF== 等非 CSV 行
    """
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return None

    try:
        # 1. 预读取：找到真正的 Header 行
        header_row_index = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # 真正的表头通常包含 "ID" 和 "Metric Name"
                if '"ID"' in line and '"Metric Name"' in line:
                    header_row_index = i
                    break
        
        print(f"Debug: Found CSV header at line {header_row_index} in {os.path.basename(filepath)}")

        # 2. 读取 CSV (跳过前面的废话)
        df = pd.read_csv(filepath, skiprows=header_row_index, on_bad_lines='skip')
        
        # 3. 清洗列名 (去除空格和引号)
        df.columns = [c.strip().replace('"', '') for c in df.columns]
        
        # 4. 检查是否成功
        required = ['ID', 'Metric Name', 'Metric Value']
        for req in required:
            if req not in df.columns:
                print(f"[Error] Column '{req}' not found after parsing.")
                print(f"Columns found: {df.columns.tolist()}")
                return None

        # 5. 数据清洗
        # 去除 Metric Value 中的千分位逗号 (例如 "1,000" -> 1000)
        df['Metric Value'] = df['Metric Value'].astype(str).str.replace(',', '')
        df['Metric Value'] = pd.to_numeric(df['Metric Value'], errors='coerce').fillna(0)

        # 6. 透视表 (转为宽格式)
        # 这样每一行就是一个 Kernel，每一列是一个指标
        df_pivot = df.pivot_table(index='ID', columns='Metric Name', values='Metric Value', aggfunc='first')
        df_pivot.fillna(0, inplace=True)
        
        return df_pivot

    except Exception as e:
        print(f"[Error] Parsing failed: {e}")
        return None

def estimate_power(df, config):
    """
    线性功耗模型
    """
    # 辅助函数：模糊匹配列名
    def get_val(keywords):
        for col in df.columns:
            # 只要列名包含关键词之一，就认为是这个指标
            for k in keywords:
                if k in col:
                    return df[col]
        return 0.0

    # 提取特征
    # 注意：这里的关键词要和你 run_mlp.sh 里写的 METRICS 对应
    insts = get_val(['sm__inst_executed', 'inst_executed'])
    mem_bytes = get_val(['dram__bytes'])
    cache_bytes = get_val(['l1tex__t_bytes', 'lts__t_sectors'])
    active_cycles = get_val(['sm__cycles_active', 'gr__cycles_active'])

    # 计算
    p_dynamic = (insts * config['W_SM']) + \
                (mem_bytes * config['W_MEM']) + \
                (cache_bytes * config['W_CACHE']) + \
                (active_cycles * config['W_ACT'])
    
    return config['P_IDLE'] + p_dynamic

def main():
    # === 请修改这里指向你的结果目录 ===
    RESULT_DIR = "/home/mark/workspace/ebpf-cupti/src/xpu/flamegraph/gpu_profiler/results_mlp"
    
    file_a = os.path.join(RESULT_DIR, "A_under_B.csv")
    file_b = os.path.join(RESULT_DIR, "B_under_A.csv")

    print(f"--- Loading Task A Data ---")
    df_a = parse_ncu_csv(file_a)
    
    print(f"--- Loading Task B Data ---")
    df_b = parse_ncu_csv(file_b)

    if df_a is None or df_b is None:
        print("Failed to load data, stopping.")
        return

    # 计算功耗序列
    print("Calculating Power Model...")
    power_a = estimate_power(df_a, MODEL_CONFIG)
    power_b = estimate_power(df_b, MODEL_CONFIG)

    # 对齐长度
    min_len = min(len(power_a), len(power_b))
    power_a = power_a.iloc[:min_len]
    power_b = power_b.iloc[:min_len]
    
    # 总功耗 = A贡献 + B贡献 - 静态功耗(扣除重复计算的Idle)
    power_total = power_a + power_b - MODEL_CONFIG['P_IDLE']

    # === 可视化 ===
    plt.figure(figsize=(10, 6))
    x = np.arange(min_len)

    plt.plot(x, power_a, 'r--', alpha=0.6, label='Task A (Compute)')
    plt.plot(x, power_b, 'b--', alpha=0.6, label='Task B (Memory)')
    
    plt.title(f"Parallel Power Sequence (Model Based)\nDir: {os.path.basename(RESULT_DIR)}")
    plt.xlabel("Kernel Sequence ID")
    plt.ylabel("Power (Watts)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_img = os.path.join(RESULT_DIR, "power_sequence.png")
    plt.savefig(out_img)
    print(f"\n[Success] Graph saved to: {out_img}")
    
    # 保存数据
    df_out = pd.DataFrame({
        'Kernel_ID': x,
        'Power_Total': power_total.values,
        'Power_A': power_a.values,
        'Power_B': power_b.values
    })
    out_csv = os.path.join(RESULT_DIR, "power_sequence.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"[Success] Data saved to: {out_csv}")

if __name__ == "__main__":
    main()