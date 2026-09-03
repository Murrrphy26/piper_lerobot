"""Forward native LeRobot image-transform config to its training CLI."""

from __future__ import annotations

import json
from typing import Any


IMAGE_TRANSFORM_FIELDS = (
    "enable",
    "max_num_transforms",
    "random_order",
    "tfs",
)


def append_image_transform_options(
    command: list[str],
    training: dict[str, Any],
) -> None:
    image_transforms = training.get("image_transforms")
    if image_transforms is None:
        return
    if not isinstance(image_transforms, dict):
        raise ValueError("training.image_transforms must be an object")

    for field in IMAGE_TRANSFORM_FIELDS:
        if field not in image_transforms:
            continue
        value = json.dumps(
            image_transforms[field],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        command.append(f"--dataset.image_transforms.{field}={value}")
