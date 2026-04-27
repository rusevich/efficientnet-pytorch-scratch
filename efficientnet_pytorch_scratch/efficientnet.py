import torch.nn as nn
import torch

from .utils import create_efficientnet_config, round_filters, round_repeats
from .model_config import ModelConfig, BlockConfig



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
                  stride=stride, groups=oup, padding=kernel_size//2, bias=False)
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
        self._blocks = nn.ModuleList() # will unpack later
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
