from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsensusGuidedPromptFeedback(nn.Module):
    """
    Consensus-Guided Prompt Feedback (CGPF).

    Functional position:
        DPCM -> OV-DCD -> ordinary LVSP -> CGPF -> DPT1

    CGPF constructs one verified object prompt for each semantic group
    from the OV-DCD-refined high-level feature x4. The prompt is then
    fed back to the ordinary-LVSP-refined middle-level feature x3.

    The design obeys four constraints:

    1. Semantic groups are processed independently.
    2. Group prompt pooling is permutation invariant.
    3. Object/background/dual consensus maps are detached before being
       used as routing variables.
    4. A bounded residual scale initialized to 1e-3 protects the
       ordinary LVSP baseline at the beginning of training.
    """

    def __init__(
        self,
        x3_channels: int = 320,
        x4_channels: int = 512,
        prompt_dim: int = 64,
        gamma_init: float = 1e-3,
        temperature: float = 0.20,
        uncertainty_floor: float = 0.50,
        min_core_support: float = 0.05,
        detach_consensus: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if x3_channels <= 0:
            raise ValueError("x3_channels must be positive")

        if x4_channels <= 0:
            raise ValueError("x4_channels must be positive")

        if prompt_dim <= 0:
            raise ValueError("prompt_dim must be positive")

        if not 0.0 < gamma_init < 1.0:
            raise ValueError("gamma_init must be in (0, 1)")

        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        if not 0.0 <= uncertainty_floor <= 1.0:
            raise ValueError(
                "uncertainty_floor must be in [0, 1]"
            )

        if min_core_support <= 0.0:
            raise ValueError(
                "min_core_support must be positive"
            )

        self.x3_channels = int(x3_channels)
        self.x4_channels = int(x4_channels)
        self.prompt_dim = int(prompt_dim)

        self.temperature = float(temperature)
        self.uncertainty_floor = float(
            uncertainty_floor
        )
        self.min_core_support = float(
            min_core_support
        )
        self.detach_consensus = bool(
            detach_consensus
        )
        self.eps = float(eps)

        # Pixel queries are extracted from ordinary-LVSP-refined x3.
        self.query_proj = nn.Conv2d(
            self.x3_channels,
            self.prompt_dim,
            kernel_size=1,
            bias=False,
        )

        # Prompt key and value are extracted from OV-DCD-refined x4.
        self.key_proj = nn.Conv2d(
            self.x4_channels,
            self.prompt_dim,
            kernel_size=1,
            bias=False,
        )

        self.value_proj = nn.Conv2d(
            self.x4_channels,
            self.x3_channels,
            kernel_size=1,
            bias=False,
        )

        # Lightweight local content branch. It does not create a new
        # decoder; it only determines what information may be added.
        self.local_refine = nn.Sequential(
            nn.Conv2d(
                self.x3_channels,
                self.x3_channels,
                kernel_size=3,
                padding=1,
                groups=self.x3_channels,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=self.x3_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.x3_channels,
                self.x3_channels,
                kernel_size=1,
                bias=False,
            ),
        )

        gamma_logit = math.log(
            gamma_init / (1.0 - gamma_init)
        )

        self.gamma_raw = nn.Parameter(
            torch.tensor(
                gamma_logit,
                dtype=torch.float32,
            )
        )

        # Runtime diagnostics. They are detached and are not part of
        # the state dict.
        self.last_gate = None
        self.last_similarity = None
        self.last_core = None
        self.last_uncertainty = None
        self.last_delta = None
        self.last_gamma = None
        self.last_group_sizes: Tuple[int, ...] = ()
        self.last_group_core_support = None

    @staticmethod
    def _normalize_group_sizes(
        batch_size: int,
        group_sizes: Optional[
            Sequence[int]
        ],
    ) -> Tuple[int, ...]:
        if group_sizes is None:
            return (int(batch_size),)

        if torch.is_tensor(group_sizes):
            values = (
                group_sizes.detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )
        else:
            values = list(group_sizes)

        normalized = tuple(
            int(value)
            for value in values
        )

        if not normalized:
            raise RuntimeError(
                "group_sizes cannot be empty"
            )

        if any(value <= 0 for value in normalized):
            raise RuntimeError(
                "All semantic group sizes must be positive"
            )

        if sum(normalized) != int(batch_size):
            raise RuntimeError(
                "Semantic group sizes do not match batch: "
                f"group_sizes={normalized}, "
                f"batch_size={batch_size}"
            )

        return normalized

    @staticmethod
    def _require_map(
        name: str,
        value: torch.Tensor,
        batch_size: int,
        spatial_size: Tuple[int, int],
    ) -> None:
        if not torch.is_tensor(value):
            raise TypeError(
                f"{name} must be a tensor"
            )

        expected = (
            int(batch_size),
            1,
            int(spatial_size[0]),
            int(spatial_size[1]),
        )

        if tuple(value.shape) != expected:
            raise RuntimeError(
                f"{name}: expected {expected}, "
                f"got {tuple(value.shape)}"
            )

        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(
                f"{name} contains NaN or Inf"
            )

    def forward(
        self,
        x3: torch.Tensor,
        x4: torch.Tensor,
        object_consensus: torch.Tensor,
        background_consensus: torch.Tensor,
        dual_attention: torch.Tensor,
        group_sizes: Optional[
            Sequence[int]
        ] = None,
    ) -> torch.Tensor:
        if x3.ndim != 4:
            raise RuntimeError(
                f"x3 must be BCHW, got {tuple(x3.shape)}"
            )

        if x4.ndim != 4:
            raise RuntimeError(
                f"x4 must be BCHW, got {tuple(x4.shape)}"
            )

        batch_size = int(x3.shape[0])

        if int(x4.shape[0]) != batch_size:
            raise RuntimeError(
                "x3/x4 batch mismatch: "
                f"{x3.shape[0]} vs {x4.shape[0]}"
            )

        if int(x3.shape[1]) != self.x3_channels:
            raise RuntimeError(
                "Unexpected x3 channels: "
                f"expected {self.x3_channels}, "
                f"got {x3.shape[1]}"
            )

        if int(x4.shape[1]) != self.x4_channels:
            raise RuntimeError(
                "Unexpected x4 channels: "
                f"expected {self.x4_channels}, "
                f"got {x4.shape[1]}"
            )

        x3_size = (
            int(x3.shape[-2]),
            int(x3.shape[-1]),
        )

        x4_size = (
            int(x4.shape[-2]),
            int(x4.shape[-1]),
        )

        if (
            x3_size[0] != 2 * x4_size[0]
            or x3_size[1] != 2 * x4_size[1]
        ):
            raise RuntimeError(
                "CGPF expects x3 to have twice the "
                "spatial resolution of x4, got "
                f"x3={x3_size}, x4={x4_size}"
            )

        self._require_map(
            "object_consensus",
            object_consensus,
            batch_size,
            x4_size,
        )

        self._require_map(
            "background_consensus",
            background_consensus,
            batch_size,
            x4_size,
        )

        self._require_map(
            "dual_attention",
            dual_attention,
            batch_size,
            x4_size,
        )

        normalized_group_sizes = (
            self._normalize_group_sizes(
                batch_size,
                group_sizes,
            )
        )

        object_map = object_consensus.clamp(
            0.0,
            1.0,
        )

        background_map = (
            background_consensus.clamp(
                0.0,
                1.0,
            )
        )

        dual_map = dual_attention.clamp(
            0.0,
            1.0,
        )

        if self.detach_consensus:
            object_map = object_map.detach()
            background_map = (
                background_map.detach()
            )
            dual_map = dual_map.detach()

        key_map = self.key_proj(x4)
        value_map = self.value_proj(x4)
        query_map = self.query_proj(x3)
        local_map = self.local_refine(x3)

        gamma = torch.sigmoid(
            self.gamma_raw
        )

        outputs = []
        gates = []
        similarities = []
        cores = []
        uncertainties = []
        deltas = []
        support_values = []

        start = 0

        for group_size in normalized_group_sizes:
            end = start + int(group_size)

            query_group = query_map[start:end]
            key_group = key_map[start:end]
            value_group = value_map[start:end]
            local_group = local_map[start:end]

            object_group = object_map[start:end]
            background_group = (
                background_map[start:end]
            )
            dual_group = dual_map[start:end]

            # Reliable object core:
            # object support with explicit background rejection.
            core = (
                object_group
                * (1.0 - background_group)
            )

            denominator = (
                core.sum(
                    dim=(0, 2, 3),
                    keepdim=True,
                )
                + self.eps
            )

            prompt_key = (
                (key_group * core)
                .sum(
                    dim=(0, 2, 3),
                    keepdim=True,
                )
                / denominator
            )

            prompt_value = (
                (value_group * core)
                .sum(
                    dim=(0, 2, 3),
                    keepdim=True,
                )
                / denominator
            )

            query_normalized = F.normalize(
                query_group,
                p=2,
                dim=1,
                eps=self.eps,
            )

            prompt_key_normalized = F.normalize(
                prompt_key,
                p=2,
                dim=1,
                eps=self.eps,
            )

            cosine_similarity = (
                query_normalized
                * prompt_key_normalized
            ).sum(
                dim=1,
                keepdim=True,
            ).clamp(
                -1.0,
                1.0,
            )

            # Smooth positive similarity gate. Unlike a hard top-k
            # operation, this remains trainable and preserves weak
            # low-contrast object regions.
            similarity_gate = torch.sigmoid(
                cosine_similarity
                / self.temperature
            )

            background_up = F.interpolate(
                background_group,
                size=x3_size,
                mode="bilinear",
                align_corners=False,
            )

            uncertainty = (
                4.0
                * dual_group
                * (1.0 - dual_group)
            )

            uncertainty_up = F.interpolate(
                uncertainty,
                size=x3_size,
                mode="bilinear",
                align_corners=False,
            )

            uncertainty_factor = (
                self.uncertainty_floor
                + (
                    1.0
                    - self.uncertainty_floor
                )
                * uncertainty_up
            )

            core_support = core.mean()

            support_scale = (
                core_support
                / self.min_core_support
            ).clamp(
                0.0,
                1.0,
            )

            # The prompt may propagate beyond the high-confidence
            # object core when x3 is semantically similar, but explicit
            # background evidence still blocks the propagation.
            spatial_gate = (
                similarity_gate
                * (1.0 - background_up)
                * uncertainty_factor
                * support_scale
            ).clamp(
                0.0,
                1.0,
            )

            prompt_value = (
                prompt_value.expand(
                    int(group_size),
                    -1,
                    x3_size[0],
                    x3_size[1],
                )
            )

            delta = (
                local_group
                + prompt_value
            )

            output_group = (
                x3[start:end]
                + gamma
                * spatial_gate
                * delta
            )

            if not bool(
                torch.isfinite(
                    output_group
                ).all()
            ):
                raise RuntimeError(
                    "CGPF produced NaN or Inf"
                )

            outputs.append(output_group)
            gates.append(spatial_gate)
            similarities.append(
                cosine_similarity
            )
            cores.append(core)
            uncertainties.append(
                uncertainty_up
            )
            deltas.append(delta)
            support_values.append(
                core_support.detach()
            )

            start = end

        output = torch.cat(
            outputs,
            dim=0,
        )

        if tuple(output.shape) != tuple(x3.shape):
            raise RuntimeError(
                "CGPF changed x3 shape: "
                f"{tuple(x3.shape)} -> "
                f"{tuple(output.shape)}"
            )

        self.last_gate = torch.cat(
            gates,
            dim=0,
        ).detach()

        self.last_similarity = torch.cat(
            similarities,
            dim=0,
        ).detach()

        self.last_core = torch.cat(
            cores,
            dim=0,
        ).detach()

        self.last_uncertainty = torch.cat(
            uncertainties,
            dim=0,
        ).detach()

        self.last_delta = torch.cat(
            deltas,
            dim=0,
        ).detach()

        self.last_gamma = gamma.detach()

        self.last_group_sizes = (
            normalized_group_sizes
        )

        self.last_group_core_support = (
            torch.stack(
                support_values
            ).detach()
        )

        return output
