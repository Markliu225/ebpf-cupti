import time
from pathlib import Path
from typing import Optional

import torch


def add_args(p) -> None:
    p.add_argument("--model", type=str, default="stabilityai/sd-turbo", help="HuggingFace model id or local path")
    p.add_argument("--prompt", type=str, default="A high-detail photo of a small robot reading a paper about GPU counters.")
    p.add_argument("--negative-prompt", type=str, default=None)
    p.add_argument("--num-images", type=int, default=1)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-inference-steps", type=int, default=1)
    p.add_argument("--guidance-scale", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--enable-safety-checker", action="store_true", help="enable safety checker (default: disabled for compatibility)")


def _load_diffusers():
    try:
        from diffusers import DiffusionPipeline
    except Exception as exc:  # pragma: no cover
        raise SystemExit("diffusers is required for sd_turbo_t2i; pip install diffusers accelerate") from exc
    return DiffusionPipeline


def _resolve_dtype(dtype: str, device: torch.device, logger) -> torch.dtype:
    if dtype == "auto":
        if device.type in {"cuda", "mps"}:
            return torch.float16
        return torch.float32

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    resolved = mapping[dtype]

    if device.type == "cpu" and resolved != torch.float32:
        logger.warning("dtype=%s on cpu may be unsupported; falling back to float32", dtype)
        return torch.float32
    if device.type == "mps" and resolved == torch.bfloat16:
        logger.warning("dtype=bfloat16 on mps may be unsupported; falling back to float16")
        return torch.float16
    return resolved


def _variant_for_dtype(dtype: torch.dtype) -> Optional[str]:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    return None


def infer(args, device: torch.device, logger) -> None:
    DiffusionPipeline = _load_diffusers()
    torch.manual_seed(int(getattr(args, "seed", 0)))

    dtype = _resolve_dtype(getattr(args, "dtype", "auto"), device, logger)
    model_id = getattr(args, "model", "stabilityai/sd-turbo")
    variant = _variant_for_dtype(dtype)

    logger.info("t2i_load model=%s dtype=%s device=%s variant=%s", model_id, dtype, device, variant)
    t0 = time.perf_counter()
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant=variant,
            use_safetensors=True,
        )
    except Exception as exc:
        logger.warning("t2i_load_retry reason=%s", str(exc).replace("\n", " ")[:4000])
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=False,
        )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    if not bool(getattr(args, "enable_safety_checker", False)) and hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
        pipe.safety_checker = None
        if hasattr(pipe, "requires_safety_checker"):
            pipe.requires_safety_checker = False
        logger.info("t2i_safety_checker disabled=true")
    load_s = time.perf_counter() - t0
    logger.info("t2i_loaded seconds=%.2f", load_s)

    prompt = getattr(args, "prompt", "")
    negative_prompt = getattr(args, "negative_prompt", None)
    num_images = max(1, int(getattr(args, "num_images", 1)))
    height = int(getattr(args, "height", 512))
    width = int(getattr(args, "width", 512))
    steps = max(1, int(getattr(args, "num_inference_steps", 1)))
    guidance = float(getattr(args, "guidance_scale", 0.0))
    generator = torch.Generator(device="cpu").manual_seed(int(getattr(args, "seed", 0)))

    t1 = time.perf_counter()
    with torch.no_grad():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            num_images_per_prompt=num_images,
            generator=generator,
        )
    gen_s = time.perf_counter() - t1

    images = getattr(out, "images", None) or []
    logger.info("t2i_generate seconds=%.2f num_images=%d size=%dx%d", gen_s, len(images), width, height)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        out_path = artifact_dir / f"{args.workflow}_{idx}.png"
        image.save(out_path)
        logger.info("artifact_saved path=%s", out_path)


def train(args, device: torch.device, logger) -> None:
    logger.info("train_mode_not_supported running_infer_instead workflow=%s", args.workflow)
    infer(args, device, logger)
