# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""ResQ utility modules."""

from .fuse_norm_utils import fuse_layer_norms, fuse_ln_linear
from .quant_utils import (
    ActQuantizer,
    ActQuantWrapper,
    WeightQuantizer,
    add_actquant,
    find_qlayers,
)
from .common import get_device, cleanup_memory, get_logger
