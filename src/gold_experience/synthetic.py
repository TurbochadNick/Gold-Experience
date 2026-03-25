from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import DishCircle


DOT_MATRIX_FONT: dict[str, list[str]] = {
    "0": [
        "01110",
        "10001",
        "10011",
        "10101",
        "11001",
        "10001",
        "01110",
    ],
    "1": [
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    "2": [
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111",
    ],
    "3": [
        "11110",
        "00001",
        "00001",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    "4": [
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010",
    ],
    "5": [
        "11111",
        "10000",
        "10000",
        "11110",
        "00001",
        "00001",
        "11110",
    ],
    "6": [
        "01110",
        "10000",
        "10000",
        "11110",
        "10001",
        "10001",
        "01110",
    ],
    "7": [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "01000",
        "01000",
    ],
    "8": [
        "01110",
        "10001",
        "10001",
        "01110",
        "10001",
        "10001",
        "01110",
    ],
    "9": [
        "01110",
        "10001",
        "10001",
        "01111",
        "00001",
        "00001",
        "01110",
    ],
}


@dataclass
class SyntheticPlate:
    image: np.ndarray
    colony_mask: np.ndarray
    label_mask: np.ndarray
    metadata: dict[str, Any]


def _random_point_in_dish(
    rng: np.random.Generator,
    dish: DishCircle,
    margin: float,
) -> tuple[int, int]:
    radius = rng.uniform(0.0, dish.radius - margin)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    x_pos = int(dish.x + radius * np.cos(angle))
    y_pos = int(dish.y + radius * np.sin(angle))
    return x_pos, y_pos


def _blend_circle(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: float,
    softness: float,
) -> None:
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1, lineType=cv2.LINE_AA)
    sigma = max(0.75, softness * radius / 3.0)
    blurred = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    weight = (blurred.astype(np.float32) / 255.0) * alpha
    for channel_index, channel_value in enumerate(color):
        base_channel = image[:, :, channel_index].astype(np.float32)
        base_channel = (1.0 - weight) * base_channel + weight * float(channel_value)
        image[:, :, channel_index] = np.clip(base_channel, 0, 255).astype(np.uint8)


def _render_label_text(
    image: np.ndarray,
    label_mask: np.ndarray,
    text: str,
    center: tuple[int, int],
    dot_radius: int,
    dot_spacing: int,
    rotation_deg: float,
    color: tuple[int, int, int],
) -> list[dict[str, Any]]:
    text = "".join(character for character in text if character in DOT_MATRIX_FONT)
    if not text:
        text = "1"

    glyph_width = 5
    glyph_height = 7
    glyph_gap = 2
    total_cols = len(text) * glyph_width + max(0, len(text) - 1) * glyph_gap
    total_rows = glyph_height
    x_offset = (total_cols - 1) * dot_spacing / 2.0
    y_offset = (total_rows - 1) * dot_spacing / 2.0

    theta = np.deg2rad(rotation_deg)
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))

    label_dots: list[dict[str, Any]] = []
    cursor_col = 0
    for character in text:
        glyph = DOT_MATRIX_FONT[character]
        for row_index, row_value in enumerate(glyph):
            for col_index, pixel in enumerate(row_value):
                if pixel != "1":
                    continue

                base_x = (cursor_col + col_index) * dot_spacing - x_offset
                base_y = row_index * dot_spacing - y_offset
                rotated_x = base_x * cos_theta - base_y * sin_theta
                rotated_y = base_x * sin_theta + base_y * cos_theta
                x_pos = int(round(center[0] + rotated_x))
                y_pos = int(round(center[1] + rotated_y))
                cv2.circle(image, (x_pos, y_pos), dot_radius, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(label_mask, (x_pos, y_pos), dot_radius + 1, 255, -1, lineType=cv2.LINE_AA)
                label_dots.append({"x": x_pos, "y": y_pos, "radius": dot_radius})
        cursor_col += glyph_width + glyph_gap

    return label_dots


def generate_plate(
    width: int = 1024,
    height: int = 1024,
    colony_count_range: tuple[int, int] = (18, 45),
    label_text: str | None = None,
    seed: int | None = None,
) -> SyntheticPlate:
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 236, dtype=np.uint8)

    y_grid, x_grid = np.mgrid[0:height, 0:width]
    gradient = ((x_grid / max(1, width - 1)) * 5.0 + (y_grid / max(1, height - 1)) * 6.0).astype(np.float32)
    image = np.clip(image.astype(np.float32) + gradient[:, :, None], 0, 255).astype(np.uint8)

    dish = DishCircle(
        x=int(width * (0.5 + rng.uniform(-0.015, 0.015))),
        y=int(height * (0.5 + rng.uniform(-0.015, 0.015))),
        radius=int(min(width, height) * rng.uniform(0.38, 0.43)),
    )

    dish_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(dish_mask, (dish.x, dish.y), dish.radius, 255, -1, lineType=cv2.LINE_AA)

    plate_color = np.array(
        [
            rng.uniform(242, 248),
            rng.uniform(244, 250),
            rng.uniform(244, 250),
        ],
        dtype=np.float32,
    )
    image[dish_mask > 0] = np.clip(plate_color, 0, 255).astype(np.uint8)

    shadow_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(shadow_mask, (dish.x + 8, dish.y + 10), dish.radius + 8, 70, -1, lineType=cv2.LINE_AA)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=12, sigmaY=12)
    shadow_strength = shadow_mask.astype(np.float32) / 255.0 * 18.0
    image = np.clip(image.astype(np.float32) - shadow_strength[:, :, None], 0, 255).astype(np.uint8)
    image[dish_mask > 0] = np.clip(image[dish_mask > 0].astype(np.float32) + 14.0, 0, 255).astype(np.uint8)

    rim_color = (210, 214, 214)
    cv2.circle(image, (dish.x, dish.y), dish.radius, rim_color, 2, lineType=cv2.LINE_AA)
    cv2.circle(image, (dish.x, dish.y), int(dish.radius * 0.96), (215, 218, 218), 2, lineType=cv2.LINE_AA)
    cv2.circle(image, (dish.x, dish.y), int(dish.radius * 0.90), (220, 223, 223), 1, lineType=cv2.LINE_AA)

    colony_mask = np.zeros((height, width), dtype=np.uint8)
    label_mask = np.zeros((height, width), dtype=np.uint8)
    colonies: list[dict[str, Any]] = []

    colony_count = int(rng.integers(colony_count_range[0], colony_count_range[1] + 1))
    for _ in range(colony_count):
        style = "sharp" if rng.random() < 0.65 else "faded"
        radius = int(rng.integers(4, 18 if style == "sharp" else 30))
        center = _random_point_in_dish(rng, dish, margin=radius + 20)
        color = (
            int(rng.integers(40, 95)),
            int(rng.integers(180, 225)),
            int(rng.integers(225, 255)),
        )
        alpha = float(rng.uniform(0.55, 0.9) if style == "sharp" else rng.uniform(0.20, 0.45))
        softness = float(rng.uniform(0.6, 1.2) if style == "sharp" else rng.uniform(1.8, 3.0))

        _blend_circle(image, center, radius, color, alpha, softness)
        cv2.circle(colony_mask, center, max(2, radius), 255, -1, lineType=cv2.LINE_AA)
        colonies.append(
            {
                "x": int(center[0]),
                "y": int(center[1]),
                "radius": int(radius),
                "style": style,
                "color_bgr": list(color),
            }
        )

    if label_text is None:
        label_text = str(int(rng.integers(1, 9999)))

    label_center = _random_point_in_dish(rng, dish, margin=110)
    label_color_choices = [
        (8, 8, 8),
        (25, 120, 25),
        (70, 80, 20),
    ]
    label_color = label_color_choices[int(rng.integers(0, len(label_color_choices)))]
    label_rotation = float(rng.uniform(-25.0, 25.0))
    label_dots = _render_label_text(
        image=image,
        label_mask=label_mask,
        text=label_text,
        center=label_center,
        dot_radius=int(rng.integers(3, 7)),
        dot_spacing=int(rng.integers(12, 18)),
        rotation_deg=label_rotation,
        color=label_color,
    )

    image = np.clip(image.astype(np.float32) + rng.normal(0.0, 1.8, size=image.shape), 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=0.7, sigmaY=0.7)

    metadata = {
        "dish": dish.to_dict(),
        "colony_count": len(colonies),
        "colonies": colonies,
        "label_text": label_text,
        "label_rotation_deg": label_rotation,
        "label_color_bgr": list(label_color),
        "label_dots": label_dots,
    }
    return SyntheticPlate(image=image, colony_mask=colony_mask, label_mask=label_mask, metadata=metadata)


def save_plate(plate: SyntheticPlate, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{stem}.png"
    colony_mask_path = output_dir / f"{stem}.colonies.png"
    label_mask_path = output_dir / f"{stem}.labels.png"
    metadata_path = output_dir / f"{stem}.meta.json"

    cv2.imwrite(str(image_path), plate.image)
    cv2.imwrite(str(colony_mask_path), plate.colony_mask)
    cv2.imwrite(str(label_mask_path), plate.label_mask)

    payload = dict(plate.metadata)
    payload["image"] = image_path.name
    payload["colony_mask"] = colony_mask_path.name
    payload["label_mask"] = label_mask_path.name
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

