import torch

import qwen3_common

DEFAULT_MODEL = "Qwen/Qwen3-4B"


def add_args(p) -> None:
    qwen3_common.add_args(p, default_model=DEFAULT_MODEL)


def train(args, device: torch.device, logger) -> None:
    qwen3_common.train(args, device, logger, default_model=DEFAULT_MODEL)


def infer(args, device: torch.device, logger) -> None:
    qwen3_common.infer(args, device, logger, default_model=DEFAULT_MODEL)

