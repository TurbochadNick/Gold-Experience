from __future__ import annotations

import cv2
import numpy as np

from .models import DishCircle


def estimate_agar_baseline(
    image: np.ndarray,
    dish: DishCircle,
    dish_mask: np.ndarray,
) -> dict[str, float]:
    """
    Sample the agar background color from the interior of the plate,
    avoiding edges where colony density and rim artifacts are higher.

    Returns median L, a, b values for the sampled agar region.
    """
    inner_mask = np.zeros_like(dish_mask)
    cv2.circle(inner_mask, (dish.x, dish.y), int(dish.radius * 0.40), 255, -1)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    pixels = lab[inner_mask == 255]
    if len(pixels) < 100:
        pixels = lab[dish_mask == 255]

    baseline_l = float(np.median(pixels[:, 0]))
    baseline_a = float(np.median(pixels[:, 1]))
    baseline_b = float(np.median(pixels[:, 2]))

    return {
        "L": baseline_l,
        "a": baseline_a,
        "b": baseline_b,
        "L_std": float(np.std(pixels[:, 0])),
    }


def apply_clahe_to_lightness(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to the lightness channel only so illumination gradients
    are reduced without directly changing chrominance.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalize_to_agar_baseline(
    image: np.ndarray,
    baseline: dict[str, float],
    target_l: float = 210.0,
    target_a: float = 128.0,
    target_b: float = 128.0,
) -> np.ndarray:
    """
    Shift LAB channels so the sampled agar baseline lands on the chosen
    canonical agar appearance. This keeps downstream thresholds operating
    in a more stable visual regime across plates.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    lab[:, :, 0] = np.clip(lab[:, :, 0] + (target_l - baseline["L"]), 0, 255)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + (target_a - baseline["a"]), 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + (target_b - baseline["b"]), 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
