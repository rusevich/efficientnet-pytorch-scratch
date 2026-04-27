from collections import namedtuple


VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8'
)

PARAMS_DICT = {
    # Coefficients:   width,depth,res,dropout
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
}

WIDTH_DIVISOR: int = 8
MIN_WIDTH: int = 8
DROP_CONNECT_RATE: float = 0.2
BATCH_NORM_MOMENTUM: float = 1 - 0.99
BATCH_NORM_EPSILON: float = 1e-3

# BLOCK_BASE holds base structure of EfficientNet-B0, BlockConfig will hold scaled version
BlockArgs = namedtuple(
    "BlockArgs",
    [
        "repeats",
        "kernel_size",
        "stride",
        "expand_ratio",
        "in_channels",
        "out_channels",
        "se_ratio",
    ],
)

BLOCKS_BASE = [
    BlockArgs(repeats=1, kernel_size=3, stride=1, expand_ratio=1, in_channels=32,  out_channels=16,  se_ratio=0.25),
    BlockArgs(repeats=2, kernel_size=3, stride=2, expand_ratio=6, in_channels=16,  out_channels=24,  se_ratio=0.25),
    BlockArgs(repeats=2, kernel_size=5, stride=2, expand_ratio=6, in_channels=24,  out_channels=40,  se_ratio=0.25),
    BlockArgs(repeats=3, kernel_size=3, stride=2, expand_ratio=6, in_channels=40,  out_channels=80,  se_ratio=0.25),
    BlockArgs(repeats=3, kernel_size=5, stride=1, expand_ratio=6, in_channels=80,  out_channels=112, se_ratio=0.25),
    BlockArgs(repeats=4, kernel_size=5, stride=2, expand_ratio=6, in_channels=112, out_channels=192, se_ratio=0.25),
    BlockArgs(repeats=1, kernel_size=3, stride=1, expand_ratio=6, in_channels=192, out_channels=320, se_ratio=0.25),
]