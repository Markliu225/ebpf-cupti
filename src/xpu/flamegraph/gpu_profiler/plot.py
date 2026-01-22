import pandas as pd
import matplotlib.pyplot as plt

def plot_power_sequence_limited(csv_file):
    # 读取数据
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{csv_file}'")
        return

    # 确保列名没有空格
    df.columns = [c.strip() for c in df.columns]

    # 获取所有唯一的 PID
    pids = df['pid'].unique()

    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 蓝、橙、绿

    for i, pid in enumerate(pids):
        # 1. 筛选出该 PID 的数据
        pid_data = df[df['pid'] == pid].copy()
        
        # 2. [关键修改] 只取前 100 个数据点
        pid_data = pid_data.head(40)
        
        # 3. 重新生成序列 ID (0, 1, 2, ... 29) 以便对齐 X 轴
        pid_data['sequence_id'] = range(len(pid_data))

        # 4. 绘制
        plt.plot(
            pid_data['sequence_id'], 
            pid_data['power_total_watts'], 
            label=f'PID {pid}',
            color=colors[i % len(colors)],
            marker='o', 
            markersize=4, 
            linewidth=2,
            alpha=0.8
        )

    # 图表美化
    plt.title('GPU Power Consumption Sequence (First 30 Kernels)', fontsize=14)
    plt.xlabel('Kernel Execution Sequence (0-30)', fontsize=12)
    plt.ylabel('Total Power (Watts)', fontsize=12)
    plt.legend(title="Process ID")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('power_sequence_top30.png')
    print("图表已保存为 'power_sequence_top30.png'")
    plt.show()

if __name__ == "__main__":
    # 确保你的当前目录下有这个文件
    plot_power_sequence_limited('/home/mark/workspace/ebpf-cupti/src/xpu/flamegraph/gpu_profiler/kernel_power_sequence.csv')