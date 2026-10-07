"""The Well's U-Net baseline, vendored so a pretrained checkpoint will load.

PolymathicAI publish trained baselines for every Well dataset, and loading one
means having the architecture it was trained with, tensor for tensor. Their
package carries it, but importing it reaches the model package's ``__init__``,
which imports their FNO and so pins ``neuraloperator==0.3.0`` -- a heavy and
exact dependency to take on for one example. This module needs torch and
nothing else.

Adapted from The Well (BSD 3-Clause), which in turn adapts PDEBench::

    the_well/benchmark/models/unet_classic/__init__.py   (the_well 1.2.0)
    https://github.com/PolymathicAI/the_well

    Takamoto et al. 2022, PDEBENCH: An Extensive Benchmark for Scientific
    Machine Learning. https://github.com/pdebench/PDEBench

Changed from the original: the Hugging Face model-hub mixin is dropped, since
the checkpoint is fetched through the dataset store like any other file; the
base class collapses into this one; and the module is typed. Layer names and
shapes are untouched, which is what lets ``state_dict`` load.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
from torch import Tensor, nn

CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
CONV_TRANSPOSE = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
POOL = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
NORM = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}


class UNetClassic(nn.Module):
    """U-Net with four down/up stages and a bottleneck.

    ``init_features`` is the width of the first stage and doubles at each one,
    so the published checkpoint's 48 gives a 768-channel bottleneck. The
    spatial resolution must divide by 16; The Well's 2D turbulent radiative
    layer is 128x384, which does.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int = 2,
        spatial_resolution: Sequence[int] = (128, 384),
        init_features: int = 48,
    ):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.spatial_resolution = tuple(spatial_resolution)
        for axis, size in enumerate(self.spatial_resolution):
            if size % 16:
                raise ValueError(
                    f"spatial axis {axis} is {size}; four poolings need a "
                    "multiple of 16"
                )

        features = init_features
        self.encoder1 = self._block(dim_in, features, "enc1")
        self.pool1 = POOL[n_spatial_dims](kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, "enc2")
        self.pool2 = POOL[n_spatial_dims](kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, "enc3")
        self.pool3 = POOL[n_spatial_dims](kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, "enc4")
        self.pool4 = POOL[n_spatial_dims](kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16, "bottleneck")

        transpose = CONV_TRANSPOSE[n_spatial_dims]
        self.upconv4 = transpose(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(features * 16, features * 8, "dec4")
        self.upconv3 = transpose(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 8, features * 4, "dec3")
        self.upconv2 = transpose(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2, "dec2")
        self.upconv1 = transpose(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, "dec1")

        self.conv = transpose(features, dim_out, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.decoder4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        dec3 = self.decoder3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.decoder2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1))
        return self.conv(dec1)

    def _block(self, in_channels: int, features: int, name: str) -> nn.Sequential:
        """Conv-norm-tanh twice. The layer names are the checkpoint's keys."""
        conv, norm = CONV[self.n_spatial_dims], NORM[self.n_spatial_dims]
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        f"{name}conv1",
                        conv(
                            in_channels, features, kernel_size=3, padding=1, bias=False
                        ),
                    ),
                    (f"{name}norm1", norm(num_features=features)),
                    (f"{name}tanh1", nn.Tanh()),
                    (
                        f"{name}conv2",
                        conv(features, features, kernel_size=3, padding=1, bias=False),
                    ),
                    (f"{name}norm2", norm(num_features=features)),
                    (f"{name}tanh2", nn.Tanh()),
                ]
            )
        )
