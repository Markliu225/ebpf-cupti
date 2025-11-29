#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
GPU_PROFILER_DIR = Path(__file__).resolve().parent
CUPTI_DIR = GPU_PROFILER_DIR / "cupti_trace"
INJECTION_SO = CUPTI_DIR / "libcupti_trace_injection.so"
CPU_PROFILER_BIN = BASE_DIR / "profiler" / "target" / "release" / "profile"

OUTPUT_DIR = BASE_DIR / "demores"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_GPU_TRACE = OUTPUT_DIR / "gpu_results.txt"
CHROME_TRACE = OUTPUT_DIR / "gpu_results.json"
CPU_TRACE = OUTPUT_DIR / "cpu_results.txt"
MERGED_TRACE = OUTPUT_DIR / "cpu_gpu_merged.folded"
TIMELINE_TRACE = OUTPUT_DIR / "cpu_gpu_timeline.txt"

sys.path.insert(0, str(BASE_DIR))
from cupti_trace_parser import CuptiTraceParser
from merge_gpu_cpu_trace import TraceMerger


def configure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    global OUTPUT_DIR, RAW_GPU_TRACE, CHROME_TRACE, CPU_TRACE, MERGED_TRACE, TIMELINE_TRACE
    OUTPUT_DIR = path
    RAW_GPU_TRACE = OUTPUT_DIR / "gpu_results.txt"
    CHROME_TRACE = OUTPUT_DIR / "gpu_results.json"
    CPU_TRACE = OUTPUT_DIR / "cpu_results.txt"
    MERGED_TRACE = OUTPUT_DIR / "cpu_gpu_merged.folded"
    TIMELINE_TRACE = OUTPUT_DIR / "cpu_gpu_timeline.txt"


def find_cuda_runtime(explicit_path: str | None = None) -> Path:
    if explicit_path:
        cuda_path = Path(explicit_path)
        if not cuda_path.exists():
            raise FileNotFoundError(f"指定的 CUDA runtime 不存在: {cuda_path}")
        return cuda_path

    candidates = [
        "/usr/local/cuda-12.9/lib64/libcudart.so.12",
        "/usr/local/cuda-13.0/lib64/libcudart.so.13",
        "/usr/local/cuda/lib64/libcudart.so.12",
        "/usr/local/cuda-12.8/lib64/libcudart.so.12",
        "/home/mark/anaconda3/envs/ebpf-cupti/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
    ]
    for path in candidates:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    raise FileNotFoundError("无法找到 libcudart.so，请通过 --cuda-lib 指定")


def start_cpu_profiler(output_path: Path, cuda_lib: str | None = None):
    if not CPU_PROFILER_BIN.exists():
        raise FileNotFoundError(
            f"未找到 CPU profiler {CPU_PROFILER_BIN}，请先在 profiler 目录运行 `cargo build --release`"
        )

    cuda_runtime = find_cuda_runtime(cuda_lib)
    cmd = [
        "sudo",
        str(CPU_PROFILER_BIN),
        "--uprobe",
        f"{cuda_runtime}:cudaLaunchKernel",
        "-E",
    ]
    out_file = open(output_path, "w")
    print(f"[CPU] 启动 eBPF profiler，输出: {output_path}")
    proc = subprocess.Popen(cmd, stdout=out_file, stderr=subprocess.PIPE)
    time.sleep(1.0)  # 等待 uprobe attach
    return proc, out_file


def stop_cpu_profiler(proc, out_file):
    if proc and proc.poll() is None:
        print("[CPU] 停止 eBPF profiler")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if out_file:
        out_file.close()


def convert_gpu_trace(raw_trace: Path, chrome_trace: Path):
    parser = CuptiTraceParser()
    events = parser.parse_file(str(raw_trace))
    parser.save_chrome_trace(
        events,
        str(chrome_trace),
        {
            "tool": "run_gpu_trace",
            "format": "Chrome Trace Format",
        },
    )
    print(f"[GPU] Chrome Trace 写入 {chrome_trace}")
    return chrome_trace


def merge_traces(cpu_trace: Path, chrome_trace: Path, output_path: Path) -> TraceMerger:
    merger = TraceMerger()
    merger.parse_cpu_trace(str(cpu_trace))
    merger.parse_gpu_trace(str(chrome_trace))
    merger.merge_traces()
    merger.write_folded_output(str(output_path))
    print(f"[MERGE] 合并结果写入 {output_path}")
    return merger


def main():
    parser = argparse.ArgumentParser(description="Minimal CPU+GPU profiler wrapper")
    parser.add_argument(
        "-c",
        "--command",
        default="cd gpu_profiler && python3 simple_dl_example.py",
        help="要执行的 Shell 命令，默认运行 simple_dl_example",
    )
    parser.add_argument("--no-cpu", action="store_true", help="仅采集 GPU，不启动 eBPF")
    parser.add_argument("--no-gpu", action="store_true", help="禁用 CUPTI 注入")
    parser.add_argument("--no-merge", action="store_true", help="不生成 CPU+GPU 对齐折叠栈")
    parser.add_argument("--pc-sampling", action="store_true", help="启用 PC 采样")
    parser.add_argument("--cuda-lib", default="/home/mark/anaconda3/envs/ebpf-cupti/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12", help="手动指定 libcudart.so 路径")
    parser.add_argument("--output-dir", help="自定义结果输出目录，默认为 repo 内的 demores")

    args = parser.parse_args()

    if args.output_dir:
        configure_output_dir(Path(args.output_dir).expanduser().resolve())

    env = os.environ.copy()
    if not args.no_gpu:
        if not INJECTION_SO.exists():
            raise FileNotFoundError(
                f"未找到 {INJECTION_SO}，请先运行 `cd {CUPTI_DIR} && make`"
            )
        env["CUDA_INJECTION64_PATH"] = str(INJECTION_SO)
        env["CUPTI_TRACE_OUTPUT_FILE"] = str(RAW_GPU_TRACE)
        if args.pc_sampling:
            env["CUPTI_ENABLE_PC_SAMPLING"] = "1"

    cpu_proc = None
    cpu_out = None
    if not args.no_cpu:
        cpu_proc, cpu_out = start_cpu_profiler(CPU_TRACE, args.cuda_lib)

    try:
        print(f"执行命令: {args.command}")
        result = subprocess.run(
            args.command,
            shell=True,
            env=env,
            cwd=str(BASE_DIR),
        )
    finally:
        if not args.no_cpu:
            stop_cpu_profiler(cpu_proc, cpu_out)

    if result.returncode != 0:
        raise RuntimeError(f"目标进程退出码 {result.returncode}")

    chrome_path = None
    if not args.no_gpu:
        chrome_path = convert_gpu_trace(RAW_GPU_TRACE, CHROME_TRACE)

    if not args.no_cpu and not args.no_gpu and not args.no_merge and chrome_path:
        merger = merge_traces(CPU_TRACE, chrome_path, MERGED_TRACE)
        merger.write_timeline(str(TIMELINE_TRACE))

    print("完成: 原始 GPU trace、CPU 报文以及可选 merged 结果均已生成。")


if __name__ == "__main__":
    main()