"""
DPT-style decoder for the frozen-DINOv2 E6 CoSOD model.

DPT1 scope:
    - decoder replacement only;
    - no boundary head;
    - no objectness decoder gate;
    - no local-detail stem;
    - no auxiliary prediction heads.

Input contract:
    x1: [B,  64, 64, 64]
    x2: [B, 128, 32, 32]
    x3: [B, 320, 16, 16]
    x4: [B, 512,  8,  8]

Output contract:
    logits: [B, 1, output_height, output_width]
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _valid_group_count(
    channels: int,
    preferred_groups: int,
) -> int:
    """Return the largest valid GroupNorm group count."""

    upper = min(
        int(preferred_groups),
        int(channels),
    )

    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups

    return 1


class ProjectionBlock(nn.Module):
    """Project one encoder feature to the shared decoder width."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()

        groups = _valid_group_count(
            out_channels,
            norm_groups,
        )

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                groups,
                out_channels,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(feature)


class ResidualConvUnit(nn.Module):
    """
    DPT-style residual convolution unit.

    GroupNorm is used instead of BatchNorm because CoSOD semantic-group
    batches have variable effective image counts.
    """

    def __init__(
        self,
        channels: int,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()

        groups = _valid_group_count(
            channels,
            norm_groups,
        )

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.norm1 = nn.GroupNorm(
            groups,
            channels,
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.norm2 = nn.GroupNorm(
            groups,
            channels,
        )

        self.activation = nn.GELU()

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        residual = feature

        feature = self.conv1(feature)
        feature = self.norm1(feature)
        feature = self.activation(feature)

        feature = self.conv2(feature)
        feature = self.norm2(feature)

        feature = feature + residual
        feature = self.activation(feature)

        return feature


class FeatureFusionBlock(nn.Module):
    """
    Fuse one projected lateral feature with the deeper decoder feature.

    This differs from the original decoder because both inputs are refined
    before and after fusion instead of being added directly.
    """

    def __init__(
        self,
        channels: int,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()

        self.lateral_refine = ResidualConvUnit(
            channels=channels,
            norm_groups=norm_groups,
        )

        self.deep_refine = ResidualConvUnit(
            channels=channels,
            norm_groups=norm_groups,
        )

        self.output_refine = ResidualConvUnit(
            channels=channels,
            norm_groups=norm_groups,
        )

    def forward(
        self,
        lateral: torch.Tensor,
        deep: torch.Tensor,
    ) -> torch.Tensor:
        lateral = self.lateral_refine(lateral)
        deep = self.deep_refine(deep)

        deep = F.interpolate(
            deep,
            size=lateral.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        fused = lateral + deep
        fused = self.output_refine(fused)

        return fused


class DPTDecoder(nn.Module):
    """Four-stage DPT-style dense prediction decoder."""

    def __init__(
        self,
        in_channels: Sequence[int] = (
            64,
            128,
            320,
            512,
        ),
        hidden_dim: int = 128,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()

        if len(in_channels) != 4:
            raise ValueError(
                "DPTDecoder expects four encoder channel values, "
                f"got {tuple(in_channels)}"
            )

        self.in_channels = tuple(
            int(value)
            for value in in_channels
        )

        self.hidden_dim = int(hidden_dim)

        self.project_x1 = ProjectionBlock(
            self.in_channels[0],
            self.hidden_dim,
            norm_groups,
        )

        self.project_x2 = ProjectionBlock(
            self.in_channels[1],
            self.hidden_dim,
            norm_groups,
        )

        self.project_x3 = ProjectionBlock(
            self.in_channels[2],
            self.hidden_dim,
            norm_groups,
        )

        self.project_x4 = ProjectionBlock(
            self.in_channels[3],
            self.hidden_dim,
            norm_groups,
        )

        self.refine_x4 = nn.Sequential(
            ResidualConvUnit(
                self.hidden_dim,
                norm_groups,
            ),
            ResidualConvUnit(
                self.hidden_dim,
                norm_groups,
            ),
        )

        self.fuse_x3 = FeatureFusionBlock(
            self.hidden_dim,
            norm_groups,
        )

        self.fuse_x2 = FeatureFusionBlock(
            self.hidden_dim,
            norm_groups,
        )

        self.fuse_x1 = FeatureFusionBlock(
            self.hidden_dim,
            norm_groups,
        )

        head_channels = max(
            self.hidden_dim // 2,
            32,
        )

        head_groups = _valid_group_count(
            head_channels,
            norm_groups,
        )

        self.prediction_head = nn.Sequential(
            ResidualConvUnit(
                self.hidden_dim,
                norm_groups,
            ),
            nn.Conv2d(
                self.hidden_dim,
                head_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                head_groups,
                head_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                head_channels,
                1,
                kernel_size=1,
            ),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Stabilize the binary saliency prediction head.
        # Keep all preceding convolution layers under Kaiming
        # initialization, but initialize the final 1x1 prediction
        # layer with a small variance to avoid saturated initial logits.
        final_conv = self.prediction_head[-1]

        if not isinstance(final_conv, nn.Conv2d):
            raise TypeError(
                "The final prediction-head module must be nn.Conv2d"
            )

        nn.init.normal_(
            final_conv.weight,
            mean=0.0,
            std=1e-3,
        )

        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    @staticmethod
    def _validate_feature(
        name: str,
        feature: torch.Tensor,
        expected_channels: int,
    ) -> None:
        if feature.ndim != 4:
            raise ValueError(
                f"{name} must be BCHW, got {tuple(feature.shape)}"
            )

        if feature.shape[1] != expected_channels:
            raise ValueError(
                f"{name} has {feature.shape[1]} channels; "
                f"expected {expected_channels}"
            )

        if not torch.isfinite(feature).all():
            raise RuntimeError(
                f"{name} contains non-finite values"
            )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        features = (
            x1,
            x2,
            x3,
            x4,
        )

        for index, (
            feature,
            expected_channels,
        ) in enumerate(
            zip(
                features,
                self.in_channels,
            ),
            start=1,
        ):
            self._validate_feature(
                name=f"x{index}",
                feature=feature,
                expected_channels=expected_channels,
            )

        r1 = self.project_x1(x1)
        r2 = self.project_x2(x2)
        r3 = self.project_x3(x3)
        r4 = self.project_x4(x4)

        y4 = self.refine_x4(r4)
        y3 = self.fuse_x3(r3, y4)
        y2 = self.fuse_x2(r2, y3)
        y1 = self.fuse_x1(r1, y2)

        logits = self.prediction_head(y1)

        logits = F.interpolate(
            logits,
            size=tuple(output_size),
            mode="bilinear",
            align_corners=False,
        )

        if logits.shape[1] != 1:
            raise RuntimeError(
                "DPT prediction head did not produce one channel"
            )

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "DPT decoder produced non-finite logits"
            )

        return logits
