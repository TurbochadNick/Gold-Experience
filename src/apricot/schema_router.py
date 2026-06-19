from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Any

import cv2
import numpy as np

from .plate_schema import SCHEMA_CLEAN_DOTS, SCHEMA_MERGED_SNOWMAN, SCHEMA_STREAK_LINES


@dataclass(frozen=True)
class SchemaRoute:
    schema: str
    confidence: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class _ComponentMetric:
    area: float
    center: tuple[float, float]
    equivalent_diameter: float
    aspect_ratio: float
    circularity: float
    extent: float


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _round_float(value: float) -> float:
    return round(float(value), 4)


def _analysis_image(image: np.ndarray, max_dim: int = 768) -> tuple[np.ndarray, float]:
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image

    height, width = bgr.shape[:2]
    longest = max(height, width, 1)
    scale = min(1.0, float(max_dim) / float(longest))
    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return bgr, scale


def _feature_mask(gray: np.ndarray) -> np.ndarray:
    min_dim = max(1, min(gray.shape[:2]))
    sigma = max(4.0, min_dim / 48.0)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    local_contrast = cv2.absdiff(gray, blurred)

    threshold = max(
        8.0,
        float(np.mean(local_contrast)) + 0.9 * float(np.std(local_contrast)),
        float(np.percentile(local_contrast, 88.0)),
    )
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[local_contrast >= threshold] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _component_metrics(mask: np.ndarray, *, min_area: float) -> list[_ComponentMetric]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    metrics: list[_ComponentMetric] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue

        x_pos, y_pos, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue

        moments = cv2.moments(contour)
        if float(moments["m00"]) == 0.0:
            center = (float(x_pos + width / 2.0), float(y_pos + height / 2.0))
        else:
            center = (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )

        rect_width, rect_height = cv2.minAreaRect(contour)[1]
        oriented_short = max(1.0, min(float(rect_width), float(rect_height)))
        oriented_long = max(float(rect_width), float(rect_height), oriented_short)
        axis_short = max(1.0, float(min(width, height)))
        axis_long = max(float(max(width, height)), axis_short)
        aspect_ratio = max(oriented_long / oriented_short, axis_long / axis_short)
        circularity = 4.0 * pi * area / (perimeter * perimeter)
        extent = area / float(width * height)
        equivalent_diameter = 2.0 * sqrt(area / pi)

        metrics.append(
            _ComponentMetric(
                area=area,
                center=center,
                equivalent_diameter=equivalent_diameter,
                aspect_ratio=float(aspect_ratio),
                circularity=float(circularity),
                extent=float(extent),
            )
        )
    return metrics


def _line_kernels(min_dim: int) -> list[np.ndarray]:
    length = int(round(min_dim / 18.0))
    length = max(13, min(43, length))
    if length % 2 == 0:
        length += 1

    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 3))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, length))
    diagonal = np.eye(length, dtype=np.uint8)
    anti_diagonal = np.fliplr(diagonal).astype(np.uint8)
    return [horizontal, vertical, diagonal, anti_diagonal]


def _combined_directional_mask(mask: np.ndarray, operation: int, kernels: list[np.ndarray]) -> np.ndarray:
    combined = np.zeros_like(mask)
    for kernel in kernels:
        transformed = cv2.morphologyEx(mask, operation, kernel, iterations=1)
        combined = cv2.bitwise_or(combined, transformed)
    return combined


def _median_spacing_ratio(metrics: list[_ComponentMetric]) -> float | None:
    round_metrics = [
        metric
        for metric in metrics
        if metric.aspect_ratio <= 1.8 and metric.circularity >= 0.45 and metric.extent >= 0.28
    ]
    if len(round_metrics) < 3:
        return None

    centers = np.array([metric.center for metric in round_metrics], dtype=float)
    deltas = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    distances[distances == 0.0] = np.inf
    nearest = np.min(distances, axis=1)
    median_nearest = float(np.median(nearest[np.isfinite(nearest)]))
    median_diameter = float(np.median([metric.equivalent_diameter for metric in round_metrics]))
    if median_diameter <= 0.0:
        return None
    return median_nearest / median_diameter


def _spacing_score(spacing_ratio: float | None) -> float:
    if spacing_ratio is None:
        return 0.45
    if spacing_ratio < 0.9:
        return _clamp(spacing_ratio / 0.9) * 0.65
    if spacing_ratio <= 12.0:
        return 1.0
    return _clamp(1.0 - (spacing_ratio - 12.0) / 18.0, 0.35, 1.0)


def route_image_schema(image: np.ndarray) -> SchemaRoute:
    analysis, scale = _analysis_image(image)
    gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    image_area = float(max(1, height * width))

    mask = _feature_mask(gray)
    min_area = max(6.0, image_area * 0.000025)
    metrics = _component_metrics(mask, min_area=min_area)
    foreground_area = float(cv2.countNonZero(mask))
    foreground_ratio = foreground_area / image_area

    elongated_metrics = [
        metric
        for metric in metrics
        if metric.aspect_ratio >= 3.6 and metric.area >= min_area * 1.5
    ]
    merged_metrics = [
        metric
        for metric in metrics
        if 1.22 <= metric.aspect_ratio <= 2.6
        and metric.circularity >= 0.50
        and metric.extent >= 0.38
        and metric.area >= min_area * 2.0
    ]
    round_metrics = [
        metric
        for metric in metrics
        if metric.aspect_ratio <= 1.8 and metric.circularity >= 0.45 and metric.extent >= 0.28
    ]

    elongated_area = float(sum(metric.area for metric in elongated_metrics))
    merged_area = float(sum(metric.area for metric in merged_metrics))
    max_aspect_ratio = max((metric.aspect_ratio for metric in metrics), default=0.0)
    component_count = len(metrics)
    round_count = len(round_metrics)
    elongated_count = len(elongated_metrics)
    merged_count = len(merged_metrics)

    kernels = _line_kernels(min(height, width))
    line_mask = _combined_directional_mask(mask, cv2.MORPH_OPEN, kernels)
    grown_mask = _combined_directional_mask(mask, cv2.MORPH_CLOSE, kernels)
    grown_metrics = _component_metrics(grown_mask, min_area=min_area)
    grown_elongated_area = float(
        sum(metric.area for metric in grown_metrics if metric.aspect_ratio >= 3.6)
    )

    line_like_area = float(cv2.countNonZero(line_mask))
    line_like_area_ratio = line_like_area / max(1.0, foreground_area)
    elongated_area_ratio = elongated_area / max(1.0, foreground_area)
    merged_area_ratio = merged_area / max(1.0, foreground_area)
    high_aspect_growth_area = max(0.0, grown_elongated_area - elongated_area)
    high_aspect_growth_area_ratio = high_aspect_growth_area / max(1.0, foreground_area)
    spacing_ratio = _median_spacing_ratio(metrics)

    round_component_ratio = round_count / max(1, component_count)
    elongated_component_ratio = elongated_count / max(1, component_count)
    merged_component_ratio = merged_count / max(1, component_count)
    area_per_component_ratio = foreground_area / max(1.0, float(component_count)) / image_area
    median_component_area_ratio = (
        float(np.median([metric.area for metric in metrics])) / image_area if metrics else 0.0
    )
    low_elongated_growth_score = 1.0 - _clamp(
        (elongated_area_ratio * 0.7 + high_aspect_growth_area_ratio * 0.5 + line_like_area_ratio * 0.4) / 0.45
    )
    low_streak_structure_score = 1.0 - _clamp(
        (elongated_area_ratio * 0.85 + high_aspect_growth_area_ratio * 0.5) / 0.42
    )

    streak_score = _clamp(
        0.36 * _clamp(elongated_area_ratio / 0.35)
        + 0.28 * _clamp(line_like_area_ratio / 0.22)
        + 0.22 * _clamp(high_aspect_growth_area_ratio / 0.50)
        + 0.10 * _clamp(max_aspect_ratio / 14.0)
        + 0.04 * _clamp(elongated_component_ratio / 0.5)
    )
    clean_score = _clamp(
        0.34 * round_component_ratio
        + 0.22 * _clamp(round_count / 12.0)
        + 0.22 * _spacing_score(spacing_ratio)
        + 0.22 * low_elongated_growth_score
    )
    merged_shape_score = _clamp(
        0.62 * merged_component_ratio
        + 0.38 * _clamp(merged_area_ratio / 0.65)
    )
    component_area_score = _clamp((area_per_component_ratio - 0.006) / 0.014)
    large_component_score = _clamp((median_component_area_ratio - 0.006) / 0.014)
    merged_score = _clamp(
        (
            0.54 * merged_shape_score
            + 0.24 * component_area_score
            + 0.22 * large_component_score
        )
        * low_streak_structure_score
    )
    if merged_count == 0:
        merged_score *= 0.35

    if foreground_area <= 0.0 or component_count == 0:
        schema = SCHEMA_CLEAN_DOTS
        confidence = 0.35
    elif streak_score >= 0.42 and streak_score > max(clean_score, merged_score) + 0.06:
        schema = SCHEMA_STREAK_LINES
        confidence = 0.50 + 0.50 * _clamp((streak_score - max(clean_score, merged_score)) / 0.65)
    elif (
        merged_score >= 0.48
        and merged_score > streak_score + 0.05
        and (merged_score > clean_score - 0.05 or merged_component_ratio >= 0.35)
    ):
        schema = SCHEMA_MERGED_SNOWMAN
        confidence = 0.50 + 0.50 * _clamp((merged_score - max(streak_score, clean_score - 0.08)) / 0.65)
    else:
        schema = SCHEMA_CLEAN_DOTS
        confidence = 0.50 + 0.50 * _clamp((clean_score - max(streak_score, merged_score)) / 0.65)

    metadata = {
        "analysis_width": int(width),
        "analysis_height": int(height),
        "analysis_scale": _round_float(scale),
        "component_count": int(component_count),
        "foreground_area_ratio": _round_float(foreground_ratio),
        "round_component_count": int(round_count),
        "round_component_ratio": _round_float(round_component_ratio),
        "elongated_component_count": int(elongated_count),
        "elongated_component_ratio": _round_float(elongated_component_ratio),
        "merged_component_count": int(merged_count),
        "merged_component_ratio": _round_float(merged_component_ratio),
        "elongated_area_ratio": _round_float(elongated_area_ratio),
        "merged_area_ratio": _round_float(merged_area_ratio),
        "line_like_area_ratio": _round_float(line_like_area_ratio),
        "high_aspect_growth_area_ratio": _round_float(high_aspect_growth_area_ratio),
        "area_per_component_ratio": _round_float(area_per_component_ratio),
        "median_component_area_ratio": _round_float(median_component_area_ratio),
        "max_component_aspect_ratio": _round_float(max_aspect_ratio),
        "median_dot_spacing_ratio": None if spacing_ratio is None else _round_float(spacing_ratio),
        "scores": {
            SCHEMA_CLEAN_DOTS: _round_float(clean_score),
            SCHEMA_MERGED_SNOWMAN: _round_float(merged_score),
            SCHEMA_STREAK_LINES: _round_float(streak_score),
        },
    }

    return SchemaRoute(
        schema=schema,
        confidence=_round_float(confidence),
        metadata=metadata,
    )
