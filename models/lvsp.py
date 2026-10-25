"""
Late-Variation Structure Preservation (LVSP).

Scope
-----
LVSP does not predict foreground/background and does not replace OV-DCD.

It uses:
1. OV-DCD uncertainty;
2. DINOv2 Block 8 -> Block 11 variation;
3. residual structure not explained by the verified x4 feature.

Only x3 is refined before DPT1 decoding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_group_count(
    channels: int,
    maximum: int = 8,
) -> int:
    for groups in range(
        min(maximum, channels),
        0,
        -1,
    ):
        if channels % groups == 0:
            return groups

    return 1


class LateVariationStructurePreservation(
    nn.Module
):
    """
    Preserve late-evolving structure inside OV-DCD-uncertain regions.

    Inputs
    ------
    x3:
        B,320,16,16

    x4:
        B,512,8,8, after DPCM and OV-DCD.

    f8, f11:
        B,384,16,16 normalized DINOv2 intermediate features.

    dual_attention:
        B,1,H,W from OV-DCD.

    Output
    ------
    refined_x3:
        B,320,16,16
    """

    def __init__(
        self,
        x3_channels: int = 320,
        x4_channels: int = 512,
        gamma_init: float = 1e-3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if not (
            0.0 < float(gamma_init) < 1.0
        ):
            raise ValueError(
                "gamma_init must be in (0, 1), "
                f"got {gamma_init}"
            )

        self.x3_channels = int(
            x3_channels
        )

        self.x4_channels = int(
            x4_channels
        )

        self.eps = float(eps)

        x3_groups = _valid_group_count(
            self.x3_channels
        )

        self.anchor_projection = nn.Sequential(
            nn.Conv2d(
                self.x4_channels,
                self.x3_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                x3_groups,
                self.x3_channels,
            ),
            nn.GELU(),
        )

        self.structure_refinement = nn.Sequential(
            nn.Conv2d(
                self.x3_channels,
                self.x3_channels,
                kernel_size=3,
                padding=1,
                groups=self.x3_channels,
                bias=False,
            ),
            nn.GroupNorm(
                x3_groups,
                self.x3_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.x3_channels,
                self.x3_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        gamma_raw_init = math.log(
            gamma_init
            / (1.0 - gamma_init)
        )

        self.gamma_raw = nn.Parameter(
            torch.tensor(
                gamma_raw_init,
                dtype=torch.float32,
            )
        )

        self.last_variation = None
        self.last_uncertainty = None
        self.last_gate = None
        self.last_residual = None
        self.last_delta = None
        self.last_gamma = None

    @staticmethod
    def _normalized_late_variation(
        f8: torch.Tensor,
        f11: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        f8_normalized = F.normalize(
            f8.float(),
            p=2.0,
            dim=1,
            eps=eps,
        )

        f11_normalized = F.normalize(
            f11.float(),
            p=2.0,
            dim=1,
            eps=eps,
        )

        return torch.linalg.vector_norm(
            f11_normalized
            - f8_normalized,
            ord=2,
            dim=1,
            keepdim=True,
        )

    @staticmethod
    def _robust_normalize(
        value: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        flat = value.flatten(1)

        lower = torch.quantile(
            flat,
            0.05,
            dim=1,
            keepdim=True,
        )

        upper = torch.quantile(
            flat,
            0.95,
            dim=1,
            keepdim=True,
        )

        normalized = (
            flat - lower
        ) / (
            upper - lower + eps
        )

        return normalized.clamp(
            0.0,
            1.0,
        ).view_as(value)

    def forward(
        self,
        x3: torch.Tensor,
        x4: torch.Tensor,
        f8: torch.Tensor,
        f11: torch.Tensor,
        dual_attention: torch.Tensor,
    ) -> torch.Tensor:
        if x3.ndim != 4:
            raise ValueError(
                "x3 must be BCHW, got "
                f"{tuple(x3.shape)}"
            )

        if x4.ndim != 4:
            raise ValueError(
                "x4 must be BCHW, got "
                f"{tuple(x4.shape)}"
            )

        if f8.ndim != 4 or f11.ndim != 4:
            raise ValueError(
                "f8/f11 must be BCHW"
            )

        if dual_attention.ndim != 4:
            raise ValueError(
                "dual_attention must be BCHW"
            )

        batch_size = x3.shape[0]

        tensors = {
            "x4": x4,
            "f8": f8,
            "f11": f11,
            "dual_attention": dual_attention,
        }

        for name, tensor in tensors.items():
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"{name}/x3 batch mismatch: "
                    f"{tensor.shape[0]} vs "
                    f"{batch_size}"
                )

        if x3.shape[1] != self.x3_channels:
            raise ValueError(
                "Unexpected x3 channels: "
                f"{x3.shape[1]} vs "
                f"{self.x3_channels}"
            )

        if x4.shape[1] != self.x4_channels:
            raise ValueError(
                "Unexpected x4 channels: "
                f"{x4.shape[1]} vs "
                f"{self.x4_channels}"
            )

        if dual_attention.shape[1] != 1:
            raise ValueError(
                "dual_attention must have "
                "one channel"
            )

        if f8.shape != f11.shape:
            raise ValueError(
                "f8/f11 shape mismatch: "
                f"{tuple(f8.shape)} vs "
                f"{tuple(f11.shape)}"
            )

        if f8.shape[-2:] != x3.shape[-2:]:
            raise ValueError(
                "f8/f11 spatial size must "
                "match x3"
            )

        variation = (
            self._normalized_late_variation(
                f8=f8,
                f11=f11,
                eps=self.eps,
            )
        )

        variation = self._robust_normalize(
            variation,
            eps=self.eps,
        )

        dual_attention = F.interpolate(
            dual_attention.float(),
            size=x3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(
            0.0,
            1.0,
        )

        uncertainty = (
            4.0
            * dual_attention
            * (1.0 - dual_attention)
        )

        # The gate is diagnostic evidence, not a new
        # foreground predictor. Stop its gradient so LVSP
        # does not alter OV-DCD verification semantics.
        gate = (
            variation * uncertainty
        ).detach()

        anchor = self.anchor_projection(
            x4
        )

        anchor = F.interpolate(
            anchor,
            size=x3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        residual = x3 - anchor

        delta = self.structure_refinement(
            residual
        )

        gamma = torch.sigmoid(
            self.gamma_raw
        )

        refined = (
            x3
            + gamma
            * gate
            * delta
        )

        if not torch.isfinite(
            refined
        ).all():
            raise RuntimeError(
                "LVSP produced non-finite values"
            )

        self.last_variation = (
            variation.detach()
        )

        self.last_uncertainty = (
            uncertainty.detach()
        )

        self.last_gate = gate.detach()

        self.last_residual = (
            residual.detach()
        )

        self.last_delta = delta.detach()

        self.last_gamma = gamma.detach()

        return refined
