from __future__ import annotations

import cv2
import numpy as np

MAX_ANALYSIS_SIDE = 1600


def downscale_for_analysis(
    image: np.ndarray,
    max_side: int = MAX_ANALYSIS_SIDE,
) -> tuple[np.ndarray, dict[str, float]]:
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image, {
            "original_width": float(width),
            "original_height": float(height),
            "analysis_width": float(width),
            "analysis_height": float(height),
            "scale": 1.0,
        }

    scale = max_side / float(longest_side)
    analysis_width = max(1, int(round(width * scale)))
    analysis_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image,
        (analysis_width, analysis_height),
        interpolation=cv2.INTER_AREA,
    )
    return resized, {
        "original_width": float(width),
        "original_height": float(height),
        "analysis_width": float(analysis_width),
        "analysis_height": float(analysis_height),
        "scale": float(scale),
    }
