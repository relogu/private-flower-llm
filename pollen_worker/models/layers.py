"""Custom layers used by some of the PFL benchmarks.

The script has been slightly modified from: TODO: add link to original script
"""

# Copyright © 2023-2024 Apple Inc.
from abc import ABC

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _NormBase  # noqa: PLC2701


class _FrozenBatchNorm(_NormBase, ABC):
    """Frozen batch normalization module.

    It will freeze the statistics during training and only update the affine parameters.
    """

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input_tensor)

        # turn of training so no batchnorm statistics is collected
        # and use pretrained statistics in training as well
        self.training = False

        exponential_average_factor = 0.0 if self.momentum is None else self.momentum

        bn_training = (self.running_mean is None) and (self.running_var is None)

        return F.batch_norm(
            input_tensor,
            (
                # If buffers are not tracked, ensure that they won't be updated
                self.running_mean
                if not self.training or self.track_running_stats
                else None
            ),
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class FrozenBatchNorm1D(_FrozenBatchNorm):
    """Frozen batch normalization module fro 1D BN."""

    def _check_input_dim(self, input_tensor: torch.Tensor) -> None:
        if input_tensor.dim() != 2 and input_tensor.dim() != 3:  # noqa: PLR2004
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class FrozenBatchNorm2D(_FrozenBatchNorm):
    """Frozen batch normalization module fro 2D BN."""

    def _check_input_dim(self, input_tensor: torch.Tensor) -> None:
        if input_tensor.dim() != 4:  # noqa: PLR2004
            raise ValueError(f"expected 4D input (got {input_tensor.dim()}D input)")


class FrozenBatchNorm3D(_FrozenBatchNorm):
    """Frozen batch normalization module fro 3D BN."""

    def _check_input_dim(self, input_tensor: torch.Tensor) -> None:
        if input_tensor.dim() != 5:  # noqa: PLR2004
            raise ValueError(f"expected 5D input (got {input_tensor.dim()}D input)")


class Transpose2D(nn.Module):
    """Transpose Tensorflow style image to PyTorch compatible."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Execute forward pass."""
        return input_tensor.permute((0, 3, 1, 2))
