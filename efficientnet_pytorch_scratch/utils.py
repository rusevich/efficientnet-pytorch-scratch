import math

from .constants import *
from .model_config import ModelConfig, BlockConfig

def round_filters(filters: int, width_coef: float) -> int:
    filters *= width_coef
    # Floor division and multiply with WIDTH_DIVISOR, to get nearest multiple
    # Add WIDTH_DIVISOR/2 to get 'ceiled' multiple for half of the time, instead of 'floored'
    new_filters = max(MIN_WIDTH, int(((filters + WIDTH_DIVISOR/2) // WIDTH_DIVISOR) * WIDTH_DIVISOR))
    if new_filters < 0.9 * filters:
        new_filters += WIDTH_DIVISOR
    return int(new_filters)

def round_repeats(repeats: int, depth_coef: float) -> int:
    return int(math.ceil(repeats * depth_coef))

def create_efficientnet_config(model_name: str, num_classes: int) -> ModelConfig:
    assert model_name in VALID_MODELS, f"Model not found, choose from {VALID_MODELS}"

    width, depth, res, dropout = PARAMS_DICT[model_name]
    blocks_configs = []
    for block in BLOCKS_BASE:
        block_config = BlockConfig(
            repeats=round_repeats(block.repeats, depth),
            kernel_size=block.kernel_size,
            stride=block.stride,
            expand_ratio=block.expand_ratio,
            in_channels=round_filters(block.in_channels, width),
            out_channels=round_filters(block.out_channels, width),
            se_ratio=block.se_ratio
        )
        blocks_configs.append(block_config)

    model_config = ModelConfig(
        model_name=model_name,
        width_coefficient=width,
        depth_coefficient=depth,
        image_size=res,
        dropout_rate=dropout,
        drop_connect_rate=DROP_CONNECT_RATE,
        batch_norm_momentum=BATCH_NORM_MOMENTUM,
        batch_norm_epsilon=BATCH_NORM_EPSILON,
        num_classes=num_classes,
        width_divisor=WIDTH_DIVISOR,
        min_width=MIN_WIDTH,
        blocks=blocks_configs
    )

    return model_config
