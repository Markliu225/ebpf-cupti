import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 功耗模型系数配置 (Config)
# ==========================================
# 这些系数代表 "每单位指标消耗的能量 (Joules)" 或 "功率权重"
# 注意：这些是经验值，用于定性分析趋势。如果是特定型号(如A100)，系数会有所不同。
POWER_MODEL = {
    # 基础静态功耗 (Watts): 只要 GPU 开启就有
    "P_STATIC_BASE": 0,  
    
    # 动态功耗权重 (Coefficients)
    # SM 指令执行 (计算功耗): 对应 sm__inst_executed_realtime.sum
    "COEFF_SM_INST": 2.0e-9, 
    
    # SM 活跃周期 (基础动态开销): 对应 sm__cycles_active.sum
    "COEFF_SM_CYCLE": 3.0e-7,
    
    # DRAM 访问 (显存读写功耗): 对应 dram__bytes.sum
    "COEFF_DRAM_BYTE": 3.0e-10,
    
    # L1/Texture Cache 访问: 对应 l1tex__t_bytes.sum
    "COEFF_L1_BYTE": 5.0e-10,
    
    # DRAM 活跃周期: 对应 dram__cycles_active.sum
    "COEFF_DRAM_CYCLE": 5.0e-7
}

def load_and_clean_data(csv_path):
    """读取并清洗 CSV 数据"""
    if not os.path.exists(csv_path):
        print(f"[Error] File not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # 填充 NaN 为 0，防止计算报错
        df.fillna(0, inplace=True)
        
        # 确保时间戳是整数 (ns)
        df['kernel_start_ns'] = df['kernel_start_ns'].astype(np.int64)
        df['kernel_end_ns'] = df['kernel_end_ns'].astype(np.int64)
        df['kernel_duration_ns'] = df['kernel_duration_ns'].astype(np.int64)
        
        # 过滤掉持续时间为 0 或异常的 Kernel
        df = df[df['kernel_duration_ns'] > 0]
        
        # 按开始时间排序
        df.sort_values(by='kernel_start_ns', inplace=True)
        
        return df
    except Exception as e:
        print(f"[Error] Failed to load CSV: {e}")
        return None

def calculate_kernel_power(df):
    """
    核心函数：计算每一行 Kernel 的平均功率
    Power (W) = Energy (J) / Time (s)
    """
    # 1. 计算该 Kernel 的总动态能量 (Dynamic Energy in Joules)
    # Energy = Sum(Count * Coeff)
    # 注意：如果 CSV 里的指标是 "Rate" (per second)，则直接算出的就是功率；
    # 但你的 CSV header 是 ".sum"，说明是总次数，所以我们要先算能量，再除以时间。
    
    dynamic_energy_joules = (
        df['sm__inst_executed_realtime.sum'] * POWER_MODEL["COEFF_SM_INST"] +
        df['sm__cycles_active.sum'] * POWER_MODEL["COEFF_SM_CYCLE"] +
        df['dram__bytes.sum'] * POWER_MODEL["COEFF_DRAM_BYTE"] +
        df.get('l1tex__t_bytes.sum', 0) * POWER_MODEL["COEFF_L1_BYTE"] +
        df.get('dram__cycles_active.sum', 0) * POWER_MODEL["COEFF_DRAM_CYCLE"]
    )
    
    # 2. 计算平均功率 (Watts)
    # Power = Energy (J) / Duration (s)
    # Duration 单位是 ns，需要除以 1e9 转为秒
    duration_seconds = df['kernel_duration_ns'] / 1e6
    
    estimated_dynamic_power = dynamic_energy_joules / duration_seconds
    
    # 3. 总功率 = 动态 + 静态
    # 注意：静态功耗是整个 GPU 共享的。如果是多任务并行，直接加静态功耗可能会重复计算。
    # 这里我们计算的是 "该 Kernel 运行期间 GPU 的总表现功耗"。
    df['power_dynamic_watts'] = estimated_dynamic_power
    df['power_total_watts'] = estimated_dynamic_power + POWER_MODEL["P_STATIC_BASE"]
    
    return df

def analyze_per_task(df):
    """按 PID 分组输出统计信息"""
    print("\n" + "="*50)
    print(" 任务功耗统计 (Per Task Analysis)")
    print("="*50)
    
    pids = df['pid'].unique()
    
    for pid in pids:
        sub_df = df[df['pid'] == pid]
        avg_power = sub_df['power_total_watts'].mean()
        max_power = sub_df['power_total_watts'].max()
        total_energy = (sub_df['power_total_watts'] * (sub_df['kernel_duration_ns'] / 1e9)).sum()
        
        comm_name = sub_df['comm'].iloc[0] if 'comm' in sub_df.columns else "Unknown"
        
        print(f"PID: {pid} ({comm_name})")
        print(f"  - Kernels Executed: {len(sub_df)}")
        print(f"  - Avg Power:        {avg_power:.2f} W")
        print(f"  - Peak Power:       {max_power:.2f} W")
        print(f"  - Est. Total Energy:{total_energy:.4f} Joules")
        print("-" * 30)

def export_results(df, output_path):
    """导出包含功耗数据的 CSV"""
    # 选一些关键列输出，避免太乱
    cols_to_save = [
        'pid', 'power_total_watts'
    ]
    
    # 确保列存在
    final_cols = [c for c in cols_to_save if c in df.columns]
    
    df[final_cols].to_csv(output_path, index=False)
    print(f"\n[Success] 估算结果已保存至: {output_path}")

def main():
    INPUT_FILE = "/home/mark/workspace/ebpf-cupti/src/xpu/flamegraph/demores_dual/kernel_metrics.csv"
    OUTPUT_FILE = "/home/mark/workspace/ebpf-cupti/src/xpu/flamegraph/gpu_profiler/kernel_power_sequence.csv"
    
    print(f"正在加载数据: {INPUT_FILE} ...")
    df = load_and_clean_data(INPUT_FILE)
    
    if df is not None:
        # 1. 计算功耗
        df = calculate_kernel_power(df)
        
        # 2. 打印每个 PID 的统计
        analyze_per_task(df)
        
        # 3. 导出结果
        export_results(df, OUTPUT_FILE)
        
        # 4. 简单的可视化提示 (Text-based Plot)
        print("\n[Preview] 前 5 个 Kernel 的功耗序列:")
        print(df[['pid', 'kernel_name', 'power_total_watts']].head().to_string())

if __name__ == "__main__":
    main()