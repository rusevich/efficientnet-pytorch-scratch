import math

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Optional, List

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
    BlockArgs(repeats=1, kernel_size=3, stride=1, expand_ratio=1, in_channels=32, out_channels=16, se_ratio=0.25),
    BlockArgs(repeats=2, kernel_size=3, stride=2, expand_ratio=6, in_channels=16, out_channels=24, se_ratio=0.25),
    BlockArgs(repeats=2, kernel_size=5, stride=2, expand_ratio=6, in_channels=24, out_channels=40, se_ratio=0.25),
    BlockArgs(repeats=3, kernel_size=3, stride=2, expand_ratio=6, in_channels=40, out_channels=80, se_ratio=0.25),
    BlockArgs(repeats=3, kernel_size=5, stride=1, expand_ratio=6, in_channels=80, out_channels=112, se_ratio=0.25),
    BlockArgs(repeats=4, kernel_size=5, stride=2, expand_ratio=6, in_channels=112, out_channels=192, se_ratio=0.25),
    BlockArgs(repeats=1, kernel_size=3, stride=1, expand_ratio=6, in_channels=192, out_channels=320, se_ratio=0.25),
]


@dataclass(frozen=True)
class BlockConfig:
    repeats: int
    kernel_size: int
    stride: int
    expand_ratio: int
    in_channels: int
    out_channels: int
    se_ratio: float


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


def round_filters(filters: int, width_coef: float) -> int:
    filters *= width_coef
    # Floor division and multiply with WIDTH_DIVISOR, to get nearest multiple
    # Add WIDTH_DIVISOR/2 to get 'ceiled' multiple for half of the time, instead of 'floored'
    new_filters = max(MIN_WIDTH, int(((filters + WIDTH_DIVISOR / 2) // WIDTH_DIVISOR) * WIDTH_DIVISOR))
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


class MBConvBlock(nn.Module):
    def __init__(self, kernel_size, stride, expand_ratio, in_channels,
                 out_channels, se_ratio, bn_mom, bn_eps):
        super().__init__()

        self.expand_ratio = expand_ratio
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # STRUCTURE:
        # 1x1 Conv Expand (if expand_ratio > 1) ->
        # Depthwise ->
        # Squeeze and Excitation ->
        # Pointwise Convolution ->
        # Skip connection and drop connect

        # 1x1 Conv expand
        inp = in_channels
        oup = in_channels * expand_ratio
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup,
                                          kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=bn_mom, eps=bn_eps)

        # Depthwise conv
        self._depthwise = nn.Conv2d(in_channels=oup, out_channels=oup, kernel_size=kernel_size,
                                    stride=stride, groups=oup, padding=kernel_size // 2, bias=False)
        # groups=oup makes every out channel dependant only on of the in channels
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=bn_mom, eps=bn_eps)

        # Squeeze and Excitation
        num_squeezed_channels = max(1, int(in_channels * se_ratio))
        self._squeeze_excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1),
            nn.Sigmoid()
        )

        # Pointwise conv
        self._pointwise = nn.Conv2d(in_channels=oup, out_channels=out_channels, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self.silu = nn.SiLU()

    def forward(self, inputs, drop_connect_rate=None):

        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self.silu(x)

        x = self._depthwise(x)
        x = self._bn1(x)
        x = self.silu(x)

        se_scale = self._squeeze_excitation(x)
        x = x * se_scale

        x = self._pointwise(x)
        x = self._bn2(x)

        # TODO drop_connect_rate

        # skip connection
        if self.stride == 1 and self.in_channels == self.out_channels:
            x += inputs

        return x


class EfficientNet(nn.Module):
    def __init__(self, config: ModelConfig, num_classes: int):
        super().__init__()
        self._config = config
        self._num_classes = num_classes

        # STRUCTURE: Stem -> MBConv Blocks -> Head -> Fully Connected

        # Stem
        stem_in_channels = 3
        stem_out_channels = round_filters(32, config.width_coefficient)
        self._stem = nn.Sequential(
            nn.Conv2d(stem_in_channels, stem_out_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=stem_out_channels, momentum=config.batch_norm_momentum,
                           eps=config.batch_norm_epsilon),
            nn.SiLU()
        )

        # Blocks
        self._blocks = nn.ModuleList()  # will unpack later
        for b in config.blocks:
            # first block increases channel count, and uses stride
            first_block = MBConvBlock(b.kernel_size, b.stride, b.expand_ratio, b.in_channels,
                                      b.out_channels, b.se_ratio, config.batch_norm_momentum, config.batch_norm_epsilon)
            self._blocks.append(first_block)
            # repeated blocks do not increase channel count
            for _ in range(b.repeats - 1):
                block = MBConvBlock(b.kernel_size, 1, b.expand_ratio, b.out_channels, b.out_channels,
                                    b.se_ratio, config.batch_norm_momentum, config.batch_norm_epsilon)
                self._blocks.append(block)

        # Head
        in_channels = config.blocks[-1].out_channels
        out_channels = round_filters(1280, config.width_coefficient)
        self._conv_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=config.batch_norm_momentum,
                                   eps=config.batch_norm_epsilon)

        # FC
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(config.dropout_rate)
        self._fc = nn.Linear(out_channels, num_classes)

        self._silu = nn.SiLU()

    @classmethod
    def from_name(cls, name: str, num_classes: int = 10) -> nn.Module:
        config = create_efficientnet_config(name, num_classes)
        model = cls(config, num_classes)
        return model

    def forward(self, inputs):
        x = inputs

        x = self._stem(x)

        for b in self._blocks:
            x = b(x)

        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._silu(x)

        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)

        return x


# USING THE EFFICIENTNET

import os

import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# random seeds

torch.manual_seed(17)
# constants

BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 1000
MAX_LEN = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    MAX_LEN = None
# installing dataset

transforms_train = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224, 224)),
    v2.Normalize(mean=(0.4625, 0.4580, 0.4295), std=(0.2798, 0.2759, 0.2988)),
])
transforms_test = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224, 224)),
    v2.Normalize(mean=(0.4601, 0.4551, 0.4275), std=(0.2784, 0.2711, 0.2876)),
])

os.makedirs("./data", exist_ok=True)
training_data = torchvision.datasets.Imagenette("data", split="train", download=True,
                                                transform=transforms_train)
test_data = torchvision.datasets.Imagenette("data", split="val", download=True,
                                            transform=transforms_test)

# dataloaders

train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = EfficientNet.from_name("efficientnet-b0", num_classes=NUM_CLASSES).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
skip_first = 0
train_loss, test_loss = [], []
train_acc, test_acc = [], []
for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_loss.append(running_loss / len(train_loader))
    train_acc.append(correct / total)

    # --- Testing ---
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = loss_fn(out, y)
            running_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    test_loss.append(running_loss / len(test_loader))
    test_acc.append(correct / total)

    print(f"Epoch {epoch + 1}: Train loss {train_loss[-1]:.4f}, Train acc {train_acc[-1]:.4f}, "
          f"Test loss {test_loss[-1]:.4f}, Test acc {test_acc[-1]:.4f}")
    if epoch > skip_first:
        plt.figure(figsize=(12, 5))
        epochs = range(1, len(train_loss[skip_first:]) + 1)

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss[skip_first:], 'b-', label='Train Loss')
        plt.plot(epochs, test_loss[skip_first:], 'r-', label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc[skip_first:], 'b-', label='Train Accuracy')
        plt.plot(epochs, test_acc[skip_first:], 'r-', label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if epoch % 20 == 0:
            plt.savefig(f"EfficientNet-B0 Imagenette {epoch} epoch")

        plt.show()
