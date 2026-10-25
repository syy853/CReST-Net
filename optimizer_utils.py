from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List

import torch
import torch.nn as nn


DINO_PREFIX = "bb.backbone."
ADAPTER_PREFIXES = (
    "bb.adapter_x1.",
    "bb.adapter_x2.",
    "bb.adapter_x3.",
    "bb.adapter_x4.",
)


def enforce_dinov2_training_contract(
    model: nn.Module,
) -> None:
    """
    Freeze only the DINOv2 foundation backbone.

    Pyramid adapters and all E6 task modules remain trainable.
    """

    for name, parameter in model.named_parameters():
        if name.startswith(DINO_PREFIX):
            parameter.requires_grad_(False)
        elif name.startswith(ADAPTER_PREFIXES):
            parameter.requires_grad_(True)

    dino_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if (
            name.startswith(DINO_PREFIX)
            and parameter.requires_grad
        )
    )

    adapter_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if (
            name.startswith(ADAPTER_PREFIXES)
            and parameter.requires_grad
        )
    )

    task_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if (
            not name.startswith("bb.")
            and parameter.requires_grad
        )
    )

    if dino_trainable != 0:
        raise RuntimeError(
            "DINOv2 backbone contains trainable parameters: "
            f"{dino_trainable}"
        )

    if adapter_trainable <= 0:
        raise RuntimeError(
            "Pyramid adapters contain no trainable parameters"
        )

    if task_trainable <= 0:
        raise RuntimeError(
            "E6 task modules contain no trainable parameters"
        )


def classify_trainable_parameters(
    model: nn.Module,
) -> Dict[str, List[nn.Parameter]]:
    groups: Dict[str, List[nn.Parameter]] = {
        "adapter": [],
        "task": [],
    }

    seen = set()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        parameter_id = id(parameter)

        if parameter_id in seen:
            raise RuntimeError(
                f"Duplicate trainable parameter: {name}"
            )

        seen.add(parameter_id)

        if name.startswith(ADAPTER_PREFIXES):
            groups["adapter"].append(parameter)
        elif name.startswith(DINO_PREFIX):
            raise RuntimeError(
                f"Frozen DINO parameter entered optimizer: {name}"
            )
        else:
            groups["task"].append(parameter)

    if not groups["adapter"]:
        raise RuntimeError(
            "Optimizer adapter group is empty"
        )

    if not groups["task"]:
        raise RuntimeError(
            "Optimizer task group is empty"
        )

    return groups


def build_optimizer(
    model: nn.Module,
    config,
) -> torch.optim.Optimizer:
    enforce_dinov2_training_contract(
        model
    )

    parameter_groups = classify_trainable_parameters(
        model
    )

    base_lr = float(
        getattr(config, "lr", 1e-4)
    )

    adapter_lr = float(
        getattr(
            config,
            "adapter_lr",
            base_lr,
        )
    )

    task_lr = float(
        getattr(
            config,
            "task_lr",
            base_lr,
        )
    )

    weight_decay = float(
        getattr(
            config,
            "weight_decay",
            1e-2,
        )
    )

    optimizer_groups = [
        {
            "params": parameter_groups["adapter"],
            "lr": adapter_lr,
            "weight_decay": weight_decay,
            "group_name": "pyramid_adapters",
        },
        {
            "params": parameter_groups["task"],
            "lr": task_lr,
            "weight_decay": weight_decay,
            "group_name": "e6_task_modules",
        },
    ]

    optimizer_name = str(
        getattr(config, "optimizer", "AdamW")
    )

    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=base_lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            optimizer_groups,
            lr=base_lr,
            weight_decay=0.0,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}"
        )

    return optimizer


def optimizer_parameter_ids(
    optimizer: torch.optim.Optimizer,
) -> set[int]:
    ids = set()

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)

            if parameter_id in ids:
                raise RuntimeError(
                    "A parameter appears in multiple optimizer groups"
                )

            ids.add(parameter_id)

    return ids


def validate_optimizer_contract(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict:
    optimizer_ids = optimizer_parameter_ids(
        optimizer
    )

    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    frozen_ids = {
        id(parameter)
        for parameter in model.parameters()
        if not parameter.requires_grad
    }

    missing = trainable_ids - optimizer_ids
    unexpected = optimizer_ids - trainable_ids
    frozen_in_optimizer = optimizer_ids & frozen_ids

    if missing:
        raise RuntimeError(
            "Trainable parameters missing from optimizer: "
            f"{len(missing)}"
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected optimizer parameters: "
            f"{len(unexpected)}"
        )

    if frozen_in_optimizer:
        raise RuntimeError(
            "Frozen parameters entered optimizer: "
            f"{len(frozen_in_optimizer)}"
        )

    return {
        "optimizer_tensors": len(optimizer_ids),
        "trainable_tensors": len(trainable_ids),
        "frozen_tensors": len(frozen_ids),
        "parameter_groups": len(
            optimizer.param_groups
        ),
    }
