from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_group_norm(channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(
                num_groups=groups,
                num_channels=channels,
            )

    raise RuntimeError(
        f"Cannot construct GroupNorm for {channels} channels"
    )


class ProjectionAdapter(nn.Module):
    """
    Lightweight trainable feature adapter.

    It performs channel projection and local spatial refinement.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        hidden_channels = max(
            out_channels // 2,
            64,
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            make_group_norm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            make_group_norm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            make_group_norm(out_channels),
            nn.GELU(),
        )

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        return self.project(feature)


class DINOv2PyramidAdapter(nn.Module):
    """
    Frozen DINOv2 ViT-S/14 plus trainable E6 pyramid adapters.

    Input
    -----
    image:
        B,3,256,256
        Already ImageNet-normalized by the original E6 dataset pipeline.

    Internal DINO input
    -------------------
    B,3,224,224

    DINO features
    -------------
    Blocks 2,5,8,11:
        B,384,16,16

    E6 outputs
    ----------
    x1: B, 64,64,64
    x2: B,128,32,32
    x3: B,320,16,16
    x4: B,512, 8, 8
    """

    OUTPUT_CHANNELS = (
        64,
        128,
        320,
        512,
    )

    OUTPUT_SIZES = (
        (64, 64),
        (32, 32),
        (16, 16),
        (8, 8),
    )

    def __init__(
        self,
        dinov2_repo: str | Path,
        checkpoint_path: str | Path,
        layer_ids: Sequence[int] = (2, 5, 8, 11),
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        self.dinov2_repo = Path(
            dinov2_repo
        ).expanduser().resolve()

        self.checkpoint_path = Path(
            checkpoint_path
        ).expanduser().resolve()

        self.layer_ids = tuple(
            int(index)
            for index in layer_ids
        )

        self.freeze_backbone = bool(
            freeze_backbone
        )

        if self.layer_ids != (2, 5, 8, 11):
            raise ValueError(
                "Initial baseline is locked to DINO layers "
                "(2, 5, 8, 11)"
            )

        if not self.dinov2_repo.is_dir():
            raise FileNotFoundError(
                self.dinov2_repo
            )

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                self.checkpoint_path
            )

        self.backbone = torch.hub.load(
            repo_or_dir=str(
                self.dinov2_repo
            ),
            model="dinov2_vits14",
            source="local",
            pretrained=False,
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        state_dict = self._extract_state_dict(
            checkpoint
        )

        incompatible = (
            self.backbone.load_state_dict(
                state_dict,
                strict=True,
            )
        )

        if (
            incompatible.missing_keys
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "Strict DINOv2 load failed: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

        if self.freeze_backbone:
            for parameter in (
                self.backbone.parameters()
            ):
                parameter.requires_grad_(False)

            self.backbone.eval()

        # Exposes the latest frozen DINO features for downstream
        # diagnostic modules without changing the backbone output contract.
        self.last_dino_features = None

        self.adapter_x1 = ProjectionAdapter(
            in_channels=384,
            out_channels=64,
        )

        self.adapter_x2 = ProjectionAdapter(
            in_channels=384,
            out_channels=128,
        )

        self.adapter_x3 = ProjectionAdapter(
            in_channels=384,
            out_channels=320,
        )

        self.adapter_x4 = ProjectionAdapter(
            in_channels=384,
            out_channels=512,
        )

        self._initialize_adapters()

    @staticmethod
    def _extract_state_dict(
        checkpoint,
    ) -> dict[str, torch.Tensor]:
        state_dict = checkpoint

        if isinstance(state_dict, dict):
            for key in (
                "state_dict",
                "model",
                "teacher",
                "student",
                "backbone",
            ):
                candidate = state_dict.get(key)

                if isinstance(candidate, dict):
                    state_dict = candidate
                    break

        if not isinstance(state_dict, dict):
            raise TypeError(
                "DINOv2 checkpoint does not contain a state dict"
            )

        cleaned = {}

        for key, value in state_dict.items():
            cleaned_key = str(key)

            changed = True

            while changed:
                changed = False

                for prefix in (
                    "module.",
                    "model.",
                    "teacher.",
                    "student.",
                    "backbone.",
                ):
                    if cleaned_key.startswith(prefix):
                        cleaned_key = cleaned_key[
                            len(prefix):
                        ]
                        changed = True

            cleaned[cleaned_key] = value

        return cleaned

    def _initialize_adapters(
        self,
    ) -> None:
        for module in (
            self.adapter_x1,
            self.adapter_x2,
            self.adapter_x3,
            self.adapter_x4,
        ):
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )

    def train(
        self,
        mode: bool = True,
    ) -> "DINOv2PyramidAdapter":
        super().train(mode)

        # Keep the frozen foundation backbone deterministic.
        if self.freeze_backbone:
            self.backbone.eval()

        return self

    def _extract_dino_features(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if image.ndim != 4:
            raise ValueError(
                "Expected BCHW image tensor, got "
                f"{tuple(image.shape)}"
            )

        if image.shape[1] != 3:
            raise ValueError(
                "Expected three input channels, got "
                f"{image.shape[1]}"
            )

        if image.shape[-2:] != (256, 256):
            raise ValueError(
                "Current E6-DINO baseline requires 256x256 input, "
                f"got {tuple(image.shape[-2:])}"
            )

        # The existing E6 loader already applies ImageNet normalization.
        # Only spatial resizing is required here.
        dino_input = F.interpolate(
            image,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        gradient_context = (
            torch.no_grad()
            if self.freeze_backbone
            else contextlib.nullcontext()
        )

        with gradient_context:
            features = (
                self.backbone.get_intermediate_layers(
                    dino_input,
                    n=self.layer_ids,
                    reshape=True,
                    return_class_token=False,
                    norm=True,
                )
            )

        if len(features) != 4:
            raise RuntimeError(
                f"Expected four DINO features, got {len(features)}"
            )

        expected_shape = (
            image.shape[0],
            384,
            16,
            16,
        )

        for layer_id, feature in zip(
            self.layer_ids,
            features,
        ):
            if tuple(feature.shape) != expected_shape:
                raise RuntimeError(
                    "Unexpected DINO feature shape at block "
                    f"{layer_id}: {tuple(feature.shape)}, "
                    f"expected {expected_shape}"
                )

            if not torch.isfinite(feature).all():
                raise RuntimeError(
                    f"Non-finite DINO feature at block {layer_id}"
                )

        return tuple(features)

    @staticmethod
    def _resize(
        feature: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        if feature.shape[-2:] == size:
            return feature

        return F.interpolate(
            feature,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        f2, f5, f8, f11 = (
            self._extract_dino_features(
                image
            )
        )

        self.last_dino_features = tuple(
            feature.detach()
            for feature in (
                f2,
                f5,
                f8,
                f11,
            )
        )

        x1 = self.adapter_x1(
            self._resize(
                f2,
                self.OUTPUT_SIZES[0],
            )
        )

        x2 = self.adapter_x2(
            self._resize(
                f5,
                self.OUTPUT_SIZES[1],
            )
        )

        x3 = self.adapter_x3(
            f8
        )

        x4 = self.adapter_x4(
            self._resize(
                f11,
                self.OUTPUT_SIZES[3],
            )
        )

        outputs = (
            x1,
            x2,
            x3,
            x4,
        )

        expected_shapes = (
            (image.shape[0], 64, 64, 64),
            (image.shape[0], 128, 32, 32),
            (image.shape[0], 320, 16, 16),
            (image.shape[0], 512, 8, 8),
        )

        for name, output, expected in zip(
            ("x1", "x2", "x3", "x4"),
            outputs,
            expected_shapes,
        ):
            if tuple(output.shape) != expected:
                raise RuntimeError(
                    f"{name} shape {tuple(output.shape)} "
                    f"does not match {expected}"
                )

            if not torch.isfinite(output).all():
                raise RuntimeError(
                    f"{name} contains NaN or Inf"
                )

        return outputs

    def backbone_parameter_count(
        self,
    ) -> int:
        return sum(
            parameter.numel()
            for parameter in self.backbone.parameters()
        )

    def backbone_trainable_parameter_count(
        self,
    ) -> int:
        return sum(
            parameter.numel()
            for parameter in self.backbone.parameters()
            if parameter.requires_grad
        )

    def adapter_trainable_parameter_count(
        self,
    ) -> int:
        return sum(
            parameter.numel()
            for name, parameter
            in self.named_parameters()
            if (
                not name.startswith("backbone.")
                and parameter.requires_grad
            )
        )
