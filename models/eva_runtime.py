import os

import torch
import torch.nn.functional as F
import timm


class FrozenEVARuntime:
    """
    External frozen EVA-02 encoder.

    This class is intentionally not an nn.Module. When stored inside
    GCoNet through object.__setattr__, EVA parameters are excluded from
    GCoNet.state_dict() and from the optimizer.
    """

    def __init__(
        self,
        checkpoint_path,
        model_name="eva02_small_patch14_224",
        input_size=224,
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.model_name = str(model_name)
        self.input_size = int(input_size)
        self._model = None
        self._device = None

        self.source_mean = (
            0.485,
            0.456,
            0.406,
        )
        self.source_std = (
            0.229,
            0.224,
            0.225,
        )
        self.eva_mean = (
            0.48145466,
            0.4578275,
            0.40821073,
        )
        self.eva_std = (
            0.26862954,
            0.26130258,
            0.27577711,
        )

    def _build(self, device):
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                "EVA-02 checkpoint not found: {}".format(
                    self.checkpoint_path
                )
            )

        model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=0,
        )

        state = torch.load(
            self.checkpoint_path,
            map_location="cpu",
        )

        if (
            isinstance(state, dict)
            and "state_dict" in state
        ):
            state = state["state_dict"]

        result = model.load_state_dict(
            state,
            strict=True,
        )

        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "Strict EVA load failed: missing={}, "
                "unexpected={}".format(
                    list(result.missing_keys),
                    list(result.unexpected_keys),
                )
            )

        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad_(False)

        model.to(device)

        self._model = model
        self._device = device

    def _ensure_device(self, device):
        if self._model is None:
            self._build(device)
            return

        if self._device != device:
            self._model.to(device)
            self._device = device

        self._model.eval()

    @staticmethod
    def _constant(values, reference):
        return reference.new_tensor(
            values
        ).view(1, 3, 1, 1)

    def _convert_normalization(self, image):
        source_mean = self._constant(
            self.source_mean,
            image,
        )
        source_std = self._constant(
            self.source_std,
            image,
        )
        eva_mean = self._constant(
            self.eva_mean,
            image,
        )
        eva_std = self._constant(
            self.eva_std,
            image,
        )

        image_01 = (
            image * source_std
            + source_mean
        ).clamp(0.0, 1.0)

        image_224 = F.interpolate(
            image_01,
            size=(
                self.input_size,
                self.input_size,
            ),
            mode="bicubic",
            align_corners=False,
        )

        return (
            image_224 - eva_mean
        ) / eva_std

    def extract(self, image):
        if image.dim() != 4:
            raise ValueError(
                "EVA input must be B,C,H,W, got {}".format(
                    tuple(image.shape)
                )
            )

        if image.shape[1] != 3:
            raise ValueError(
                "EVA input must contain three channels."
            )

        self._ensure_device(image.device)

        eva_input = self._convert_normalization(
            image
        )

        with torch.no_grad():
            tokens = self._model.forward_features(
                eva_input
            )

        if tokens.dim() != 3:
            raise RuntimeError(
                "Expected EVA tokens B,N,C, got {}".format(
                    tuple(tokens.shape)
                )
            )

        prefix_tokens = int(
            getattr(
                self._model,
                "num_prefix_tokens",
                1,
            )
        )

        patch_tokens = tokens[
            :,
            prefix_tokens:,
            :,
        ]

        batch_size, token_count, channels = (
            patch_tokens.shape
        )

        side = int(token_count ** 0.5)

        if side * side != token_count:
            raise RuntimeError(
                "EVA patch tokens cannot form a square: "
                "{}".format(token_count)
            )

        feature_map = (
            patch_tokens
            .transpose(1, 2)
            .reshape(
                batch_size,
                channels,
                side,
                side,
            )
            .contiguous()
        )

        if not torch.isfinite(feature_map).all():
            raise RuntimeError(
                "EVA feature map contains NaN or Inf."
            )

        return feature_map
