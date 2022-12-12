from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvalidKernelSize(Exception):
    pass


class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size):
        super().__init__()
        if kernel_size % 2 != 1:
            raise InvalidKernelSize("Conv kernel size should be odd number")
        padding = (kernel_size - 1) // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownStep(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, out_channels, conv_kernel_size),
        )

    def forward(self, x):
        return self.layers(x)


class UpStep(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size,
        conv_transpose_kernel_size,
    ):
        super().__init__()
        if conv_transpose_kernel_size % 2 != 0:
            raise InvalidKernelSize("Conv-transpose kernel size should be even number")
        padding = (conv_transpose_kernel_size - 2) // 2
        self.upscale = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=conv_transpose_kernel_size,
            stride=2,
            padding=padding,
        )
        self.double_conv = DoubleConv(
            in_channels, out_channels, out_channels, conv_kernel_size
        )

    def forward(self, x1, x2):
        return self.double_conv(torch.cat([x2, self.upscale(x1)], dim=1))


@dataclass
class UNetConfigs:
    in_channels: int  # Number of model input image channels (RGB -> 3)
    out_channels: int  # Number of model output channels (n classes)
    hidden_channels_scale: int  # determine number of channels in hidden layers
    steps: int  # determine number of down and upsample steps
    conv_kernel_size: int  # should be odd number
    conv_transpose_kernel_size: int  # should be even number


class UNet(nn.Module):
    def __init__(self, configs: UNetConfigs):
        super().__init__()
        self.configs = configs

        self.initial_conv = DoubleConv(
            self.configs.in_channels,
            2**self.configs.hidden_channels_scale,
            2**self.configs.hidden_channels_scale,
            self.configs.conv_kernel_size,
        )
        self.down_steps = nn.ModuleList()
        for i in range(configs.steps):
            self.down_steps.append(
                DownStep(
                    2 ** (self.configs.hidden_channels_scale + i),
                    2 ** (self.configs.hidden_channels_scale + i + 1),
                    self.configs.conv_kernel_size,
                )
            )

        self.up_steps = nn.ModuleList()
        for i in range(configs.steps, 0, -1):
            self.up_steps.append(
                UpStep(
                    2 ** (self.configs.hidden_channels_scale + i),
                    2 ** (self.configs.hidden_channels_scale + i - 1),
                    self.configs.conv_kernel_size,
                    self.configs.conv_transpose_kernel_size,
                )
            )
        self.final_conv = nn.Conv2d(
            2**self.configs.hidden_channels_scale,
            self.configs.out_channels,
            kernel_size=1,
        )

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters()) * 1e-6

    @property
    def size_on_disk(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def forward(self, x):
        x = self.initial_conv(x)
        cache = [x]
        for m in self.down_steps:
            x = m(x)
            cache.append(x)
        cache.pop()  # Final down step output is not needed as skip connection
        for m in self.up_steps:
            x = m(x, cache.pop())
        x = self.final_conv(x)
        return x
