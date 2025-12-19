import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch


def select_device(preferred: Optional[str] = None) -> torch.device:
    """Choose an available torch device with a CUDA-first policy.

    preferred: optional string in {"cuda", "mps", "cpu"}
    """

    if preferred:
        preferred = preferred.lower()
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if preferred == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logger(log_dir: Path, workflow: str, mode: str, device: torch.device) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{workflow}_{mode}_{ts}.log"

    logger = logging.getLogger(f"{workflow}_{mode}_{ts}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    logger.info("run_start workflow=%s mode=%s device=%s log_file=%s", workflow, mode, device, log_path)
    return logger


def log_complete(logger: logging.Logger, start_time: float, extra: Optional[Dict[str, str]] = None) -> None:
    duration = time.time() - start_time
    parts = [f"duration_s={duration:.2f}"]
    if extra:
        parts.extend(f"{k}={v}" for k, v in extra.items())
    logger.info("run_complete %s", " ".join(parts))


def prep_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def recommended_num_workers() -> int:
    # Avoid too many workers on small systems; MPS benefits less from pinning.
    try:
        import multiprocessing as mp

        return max(2, min(4, mp.cpu_count() // 2))
    except Exception:
        return 2

