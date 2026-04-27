from dataclasses import dataclass, field
from typing import Optional, List


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    width_coefficient: float
    depth_coefficient: float
    image_size: int
    dropout_rate: float

    drop_connect_rate: float
    batch_norm_momentum: float
    batch_norm_epsilon: float
    num_classes: int
    width_divisor: int
    min_width: int
    blocks: List[BlockConfig] = field(default_factory=list)


@dataclass(frozen=True)
class BlockConfig:
    repeats: int
    kernel_size: int
    stride: int
    expand_ratio: int
    in_channels: int
    out_channels: int
    se_ratio: float