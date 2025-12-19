import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

from workflow import (
    agnews_textcnn,
    cifar_resnet,
    convnext_benchmark,
    dcnv2_ctr,
    jena_lstm,
    movielens_mf,
    qwen3_0p6b,
    qwen3_1p7b,
    qwen3_4b,
    sd_turbo_t2i,
    shakespeare_gpt,
    synthetic_embedding_dlrm,
    synthetic_gemm_fp32,
    synthetic_gemm_tensor,
    synthetic_resnet50,
    synthetic_transformer_encoder,
)
from workflow.common import log_complete, prep_path, select_device, setup_logger


@dataclass
class Workflow:
    description: str
    add_args: Callable[[argparse.ArgumentParser], None]
    train_fn: Callable
    infer_fn: Callable


def add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["train", "infer"], default="train")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--log-dir", type=Path, default=Path("logs"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--limit-batches", type=int, default=None, help="optional limit on batches per epoch for quick runs")
    p.add_argument("--checkpoint", type=Path, default=None, help="path to a checkpoint for inference; defaults to artifact path")


WORKFLOWS: Dict[str, Workflow] = {
    "cifar_resnet": Workflow(
        description="Vision classification on CIFAR10 with ResNet18",
        add_args=cifar_resnet.add_args,
        train_fn=cifar_resnet.train,
        infer_fn=cifar_resnet.infer,
    ),
    "agnews_textcnn": Workflow(
        description="Text classification on AG_NEWS with TextCNN",
        add_args=agnews_textcnn.add_args,
        train_fn=agnews_textcnn.train,
        infer_fn=agnews_textcnn.infer,
    ),
    "shakespeare_gpt": Workflow(
        description="Character-level language model on Tiny Shakespeare",
        add_args=shakespeare_gpt.add_args,
        train_fn=shakespeare_gpt.train,
        infer_fn=shakespeare_gpt.infer,
    ),
    "movielens_mf": Workflow(
        description="Matrix factorization recommender on MovieLens 100k",
        add_args=movielens_mf.add_args,
        train_fn=movielens_mf.train,
        infer_fn=movielens_mf.infer,
    ),
    "jena_lstm": Workflow(
        description="Time-series forecasting with LSTM on Jena climate",
        add_args=jena_lstm.add_args,
        train_fn=jena_lstm.train,
        infer_fn=jena_lstm.infer,
    ),
    "synthetic_gemm_fp32": Workflow(
        description="Synthetic FP32 GEMM/MLP workload (compute-bound)",
        add_args=synthetic_gemm_fp32.add_args,
        train_fn=synthetic_gemm_fp32.train,
        infer_fn=synthetic_gemm_fp32.infer,
    ),
    "synthetic_gemm_tensor": Workflow(
        description="Synthetic AMP GEMM/MLP workload (tensor/AMP, compute-bound)",
        add_args=synthetic_gemm_tensor.add_args,
        train_fn=synthetic_gemm_tensor.train,
        infer_fn=synthetic_gemm_tensor.infer,
    ),
    "synthetic_embedding_dlrm": Workflow(
        description="Synthetic embedding-heavy recommender (memory-bound)",
        add_args=synthetic_embedding_dlrm.add_args,
        train_fn=synthetic_embedding_dlrm.train,
        infer_fn=synthetic_embedding_dlrm.infer,
    ),
    "synthetic_resnet50": Workflow(
        description="Synthetic ResNet50 training/inference workload (vision)",
        add_args=synthetic_resnet50.add_args,
        train_fn=synthetic_resnet50.train,
        infer_fn=synthetic_resnet50.infer,
    ),
    "synthetic_transformer_encoder": Workflow(
        description="Synthetic Transformer Encoder workload (attention + FFN)",
        add_args=synthetic_transformer_encoder.add_args,
        train_fn=synthetic_transformer_encoder.train,
        infer_fn=synthetic_transformer_encoder.infer,
    ),
    "qwen3_0p6b": Workflow(
        description="LLM inference (Qwen3 0.6B, text generation)",
        add_args=qwen3_0p6b.add_args,
        train_fn=qwen3_0p6b.train,
        infer_fn=qwen3_0p6b.infer,
    ),
    "qwen3_1p7b": Workflow(
        description="LLM inference (Qwen3 1.7B, text generation)",
        add_args=qwen3_1p7b.add_args,
        train_fn=qwen3_1p7b.train,
        infer_fn=qwen3_1p7b.infer,
    ),
    "qwen3_4b": Workflow(
        description="LLM inference (Qwen3 4B, text generation)",
        add_args=qwen3_4b.add_args,
        train_fn=qwen3_4b.train,
        infer_fn=qwen3_4b.infer,
    ),
    "sd_turbo_t2i": Workflow(
        description="Text-to-image generation (Stable Diffusion Turbo)",
        add_args=sd_turbo_t2i.add_args,
        train_fn=sd_turbo_t2i.train,
        infer_fn=sd_turbo_t2i.infer,
    ),
    "convnext_benchmark": Workflow(
        description="Synthetic ConvNeXt benchmark (vision)",
        add_args=convnext_benchmark.add_args,
        train_fn=convnext_benchmark.train,
        infer_fn=convnext_benchmark.infer,
    ),
    "dcnv2_ctr": Workflow(
        description="Synthetic CTR model (DCNv2-style cross + MLP)",
        add_args=dcnv2_ctr.add_args,
        train_fn=dcnv2_ctr.train,
        infer_fn=dcnv2_ctr.infer,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified AI workflow runner (PyTorch)")
    subparsers = parser.add_subparsers(dest="workflow", required=True)
    for name, wf in WORKFLOWS.items():
        sp = subparsers.add_parser(name, help=wf.description)
        add_shared_args(sp)
        wf.add_args(sp)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = select_device(args.device)
    prep_path(args.data_root)
    prep_path(args.artifact_dir)
    prep_path(args.log_dir)
    logger = setup_logger(args.log_dir, args.workflow, args.mode, device)
    start_time = time.time()

    wf = WORKFLOWS[args.workflow]
    if args.mode == "train":
        wf.train_fn(args, device, logger)
    else:
        wf.infer_fn(args, device, logger)

    log_complete(logger, start_time, {"workflow": args.workflow, "mode": args.mode})


if __name__ == "__main__":
    main()
