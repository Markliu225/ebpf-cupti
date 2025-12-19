import time
from pathlib import Path
from typing import List, Optional

import torch


def add_args(p, default_model: str) -> None:
    p.add_argument("--model", type=str, default=default_model, help="HuggingFace model id or local path")
    p.add_argument("--prompt", type=str, default="Write a short paragraph about GPU performance counters.")
    p.add_argument("--num-prompts", type=int, default=1, help="number of prompts in a batch (LLM ignores --batch-size)")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--do-sample", action="store_true", help="use sampling (default: greedy decoding)")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--trust-remote-code", action="store_true", help="allow loading custom model code from the repo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-runtime-sec", type=float, default=15.0, help="ensure run lasts long enough for sampling")


def _resolve_dtype(dtype: str, device: torch.device, logger) -> torch.dtype:
    if dtype == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    resolved = mapping[dtype]

    if device.type == "cpu" and resolved == torch.float16:
        logger.warning("dtype=float16 on cpu is often unsupported/slow; falling back to float32")
        return torch.float32
    if device.type == "mps" and resolved == torch.bfloat16:
        logger.warning("dtype=bfloat16 on mps may be unsupported; falling back to float16")
        return torch.float16
    return resolved


def _load_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise SystemExit("transformers is required for qwen3 workflows; pip install transformers") from exc
    return AutoModelForCausalLM, AutoTokenizer


def _build_prompts(prompt: str, num_prompts: int) -> List[str]:
    num_prompts = max(1, int(num_prompts))
    return [prompt] * num_prompts


def infer(args, device: torch.device, logger, default_model: str) -> None:
    start_time = time.time()
    AutoModelForCausalLM, AutoTokenizer = _load_transformers()

    model_id = getattr(args, "model", None) or default_model
    torch.manual_seed(int(getattr(args, "seed", 0)))
    dtype = _resolve_dtype(getattr(args, "dtype", "auto"), device, logger)

    logger.info("llm_load model=%s dtype=%s device=%s", model_id, dtype, device)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=bool(getattr(args, "trust_remote_code", False)))
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
    )
    model.to(device)
    model.eval()
    load_s = time.perf_counter() - t0
    logger.info("llm_loaded seconds=%.2f", load_s)

    prompts = _build_prompts(getattr(args, "prompt", ""), getattr(args, "num_prompts", 1))
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = bool(getattr(args, "do_sample", False))
    gen_kwargs = {
        "max_new_tokens": int(getattr(args, "max_new_tokens", 64)),
        "do_sample": do_sample,
        "temperature": float(getattr(args, "temperature", 0.8)),
        "top_p": float(getattr(args, "top_p", 0.95)),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if not do_sample:
        gen_kwargs.pop("temperature", None)
        gen_kwargs.pop("top_p", None)

    t1 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    gen_s = time.perf_counter() - t1

    texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    sample = texts[0] if texts else ""
    sample_preview = sample.replace("\n", "\\n")[:4000]
    logger.info("llm_generate seconds=%.2f num_prompts=%d preview=%s", gen_s, len(prompts), sample_preview)

    out_path = Path(args.artifact_dir) / f"{args.workflow}_generated.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(texts), encoding="utf-8")
    logger.info("artifact_saved path=%s", out_path)
    _ensure_min_runtime(start_time, getattr(args, "min_runtime_sec", 0.0), device)


def train(args, device: torch.device, logger, default_model: str) -> None:
    logger.info("train_mode_not_supported running_infer_instead workflow=%s", args.workflow)
    infer(args, device, logger, default_model=default_model)


def _ensure_min_runtime(start_time: float, min_runtime_sec: float, device: torch.device) -> None:
    if min_runtime_sec <= 0:
        return
    a = torch.randn(2048, 1024, device=device)
    b = torch.randn(1024, 2048, device=device)
    while True:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.time() - start_time
        if elapsed >= min_runtime_sec:
            break
        torch.matmul(a, b)
