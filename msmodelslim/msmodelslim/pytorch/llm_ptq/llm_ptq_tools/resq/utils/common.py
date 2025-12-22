# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ
"""Common utilities for ResQ."""

import gc
import logging
import os
from typing import Optional

import torch

# Check for NPU availability
npu_available = False
try:
    import torch_npu
    npu_available = torch.npu.is_available()
except ImportError:
    pass


def get_device():
    """Get the appropriate device (NPU, CUDA, or CPU)."""
    if npu_available:
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEV = get_device()


def cleanup_memory(verbos=True):
    """Run GC and clear GPU/NPU memory."""
    import inspect

    caller_name = ""
    try:
        caller_name = f" (from {inspect.stack()[1].function})"
    except (ValueError, KeyError):
        pass

    def total_reserved_mem():
        if npu_available:
            return sum(
                torch.npu.memory_reserved(device=i)
                for i in range(torch.npu.device_count())
            )
        elif torch.cuda.is_available():
            return sum(
                torch.cuda.memory_reserved(device=i)
                for i in range(torch.cuda.device_count())
            )
        return 0

    memory_before = total_reserved_mem()

    gc.collect()

    if npu_available:
        torch.npu.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"NPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> "
                f"{memory_after / (1024 ** 3):.2f} GB "
                f"({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> "
                f"{memory_after / (1024 ** 3):.2f} GB "
                f"({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def get_logger(
    logger_name: Optional[str], log_file_name: Optional[str] = None
) -> logging.Logger:
    """Get a configured logger."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file_name:
        file_handler = logging.FileHandler(log_file_name)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_local_rank() -> int:
    """Get local rank for distributed training."""
    if os.environ.get("LOCAL_RANK"):
        return int(os.environ["LOCAL_RANK"])
    else:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if npu_available:
        torch.npu.manual_seed_all(seed)
