# Adapted from xuhongzuo/deep-iForest, algorithms/net_torch.py.
# Original author: Xu Hongzuo (hongzuo.xu@gmail.com), 2022/8/19.
# Apache-2.0; see ../LICENSE and ../NOTICE.
# Modified: retain only the TS backbone and read its settings from config;
# omit graph dependencies and the unused dropout module. No optimization.

import torch
from torch import nn
from torch.nn import functional as F


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        receptive_field = (kernel_size - 1) * dilation + 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=receptive_field // 2, dilation=dilation,
        )
        self.remove = 1 if receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.remove] if self.remove else out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation)
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final else None
        )

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = self.conv1(F.gelu(x))
        x = self.conv2(F.gelu(x))
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, n_features, *, hidden_dim, rep_dim, layers, kernel_size,
                 dilation_base, activation, pooling, dropout):
        super().__init__()
        if activation != "gelu" or pooling != "max" or dropout != 0:
            raise ValueError("The contracted CNN uses GELU, max pooling and no dropout.")
        self.input_fc = nn.Linear(n_features, hidden_dim)
        channels = [hidden_dim] * layers + [rep_dim]
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else hidden_dim,
                channels[i], kernel_size, dilation_base ** i,
                final=(i == len(channels) - 1),
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        x = self.input_fc(x).transpose(1, 2)
        x = self.net(x)
        return F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
