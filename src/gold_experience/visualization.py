from __future__ import annotations

import cv2
import numpy as np

from .models import PipelineResult


def render_overlay(
    image: np.ndarray,
    result: PipelineResult,
    show_rejected: bool = True,
) -> np.ndarray:
    overlay = image.copy()
    cv2.circle(
        overlay,
        (result.dish.x, result.dish.y),
        result.dish.radius,
        (255, 210, 0),
        2,
        lineType=cv2.LINE_AA,
    )

    label_ids = set(result.label_ids)
    colony_ids = set(result.colony_ids)

    for candidate in result.candidates:
        if candidate.id in colony_ids:
            color = (0, 210, 0)
            thickness = 2
        elif candidate.id in label_ids:
            color = (0, 0, 255)
            thickness = 2
        elif show_rejected:
            color = (0, 215, 255)
            thickness = 1
        else:
            continue

        center = (int(round(candidate.center[0])), int(round(candidate.center[1])))
        radius = max(4, int(round(candidate.equivalent_radius)))
        cv2.circle(overlay, center, radius, color, thickness, lineType=cv2.LINE_AA)

    cv2.putText(
        overlay,
        f"Colonies: {result.colony_count}",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"Labels: {len(result.label_ids)}",
        (24, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        lineType=cv2.LINE_AA,
    )
    return overlay

