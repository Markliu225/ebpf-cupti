import subprocess
import sys
import time
import os

# === 配置 ===
# 使用当前环境的 Python 解析器
PYTHON_EXEC = sys.executable 

def main():
    print(f"[Launcher] Starting process tree with Python: {PYTHON_EXEC}")

    # 1. 启动任务 A (计算) - 子进程 1
    # 注意：我们让它跑 measure 模式，捕捉序列
    cmd_a = [PYTHON_EXEC, "task_a_compute.py", "--mode", "measure", "--iters", "50"]
    print(f"[Launcher] Spawning Task A: {' '.join(cmd_a)}")
    proc_a = subprocess.Popen(cmd_a)

    # 2. 启动任务 B (访存) - 子进程 2
    cmd_b = [PYTHON_EXEC, "task_b_memory.py", "--mode", "measure", "--iters", "50"]
    print(f"[Launcher] Spawning Task B: {' '.join(cmd_b)}")
    proc_b = subprocess.Popen(cmd_b)

    # 3. 等待两个任务完成
    # ncu 会一直监控，直到这两个子进程都结束
    exit_code_a = proc_a.wait()
    exit_code_b = proc_b.wait()

    print(f"[Launcher] All tasks finished. A: {exit_code_a}, B: {exit_code_b}")

if __name__ == "__main__":
    main()