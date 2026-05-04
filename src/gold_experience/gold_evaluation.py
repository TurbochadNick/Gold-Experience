from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .annotations import PlateGoldAnnotation, PolygonAnnotation
from .image_preprocessing import downscale_for_analysis
from .pipeline import ColonyCounterPipeline


@dataclass(frozen=True)
class PointMatch:
    prediction_index: int
    truth_index: int
    distance: float


@dataclass(frozen=True)
class TruthColony:
    x: float
    y: float
    tolerance: float
    morphology: str


def _scaled_points(points: list[tuple[float, float]], scale: float) -> list[tuple[float, float]]:
    return [(x_pos * scale, y_pos * scale) for x_pos, y_pos in points]


def _scaled_polygon(polygon: PolygonAnnotation, scale: float) -> list[tuple[float, float]]:
    return _scaled_points(polygon.points, scale)


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    contour = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, point, False) >= 0


def _point_in_any_polygon(
    point: tuple[float, float],
    polygons: list[list[tuple[float, float]]],
) -> bool:
    return any(_point_in_polygon(point, polygon) for polygon in polygons)


def _match_points(
    predictions: list[tuple[float, float]],
    truth: list[TruthColony],
) -> list[PointMatch]:
    pairs: list[PointMatch] = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, true_point in enumerate(truth):
            distance = hypot(prediction[0] - true_point.x, prediction[1] - true_point.y)
            if distance <= true_point.tolerance:
                pairs.append(
                    PointMatch(
                        prediction_index=prediction_index,
                        truth_index=truth_index,
                        distance=distance,
                    )
                )

    pairs.sort(key=lambda item: item.distance)
    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    matches: list[PointMatch] = []
    for pair in pairs:
        if pair.prediction_index in used_predictions or pair.truth_index in used_truth:
            continue
        used_predictions.add(pair.prediction_index)
        used_truth.add(pair.truth_index)
        matches.append(pair)
    return matches


def _dish_error(result_dish: Any, gold: PlateGoldAnnotation, scale: float) -> dict[str, float] | None:
    if gold.dish is None:
        return None
    gold_x = gold.dish.cx * scale
    gold_y = gold.dish.cy * scale
    gold_radius = ((gold.dish.rx + gold.dish.ry) / 2.0) * scale
    return {
        "center_error": float(hypot(result_dish.x - gold_x, result_dish.y - gold_y)),
        "radius_error": float(abs(result_dish.radius - gold_radius)),
    }


def _truth_colonies(gold: PlateGoldAnnotation, scale: float, base_tolerance: float) -> list[TruthColony]:
    truth: list[TruthColony] = []
    for item in gold.colonies:
        morphology = item.morphology
        radius = 0.0 if item.radius is None else item.radius * scale
        if morphology == "ellipse":
            # Blob annotations describe an approximate object footprint, not a
            # pinpoint center. Matching should allow a detector center anywhere
            # reasonably near the annotated blob center without swallowing
            # adjacent colonies in dense plates.
            tolerance = max(base_tolerance, min(48.0, radius * 0.75))
        else:
            tolerance = base_tolerance
        truth.append(
            TruthColony(
                x=item.x * scale,
                y=item.y * scale,
                tolerance=tolerance,
                morphology=morphology,
            )
        )
    return truth


def evaluate_gold_image(
    image_path: Path,
    annotation_path: Path,
    pipeline: ColonyCounterPipeline,
    match_tolerance: float = 18.0,
) -> dict[str, Any]:
    gold = PlateGoldAnnotation.load(annotation_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    analysis_image, resize_info = downscale_for_analysis(image)
    scale = float(resize_info["scale"])
    result = pipeline.run(analysis_image)
    prediction_ids = set(result.colony_ids)
    predictions = [
        tuple(candidate.center)
        for candidate in result.candidates
        if candidate.id in prediction_ids
    ]
    truth = _truth_colonies(gold, scale=scale, base_tolerance=match_tolerance)
    matches = _match_points(predictions, truth)
    matched_predictions = {item.prediction_index for item in matches}

    scaled_label_regions = [_scaled_polygon(polygon, scale) for polygon in gold.label_regions]
    false_positive_points = [
        point
        for index, point in enumerate(predictions)
        if index not in matched_predictions
    ]
    label_region_false_positives = sum(
        1 for point in false_positive_points if _point_in_any_polygon(point, scaled_label_regions)
    )

    true_positive = len(matches)
    false_positive = len(predictions) - true_positive
    false_negative = len(truth) - true_positive
    matched_truth = {item.truth_index for item in matches}
    truth_point_count = sum(1 for item in truth if item.morphology == "point")
    truth_ellipse_count = sum(1 for item in truth if item.morphology == "ellipse")
    point_true_positive = sum(
        1
        for index, item in enumerate(truth)
        if item.morphology == "point" and index in matched_truth
    )
    ellipse_true_positive = sum(
        1
        for index, item in enumerate(truth)
        if item.morphology == "ellipse" and index in matched_truth
    )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)

    return {
        "image": gold.image,
        "image_path": str(image_path),
        "annotation_path": str(annotation_path),
        "original_size": [
            int(resize_info["original_width"]),
            int(resize_info["original_height"]),
        ],
        "analysis_size": [
            int(resize_info["analysis_width"]),
            int(resize_info["analysis_height"]),
        ],
        "scale": scale,
        "candidate_count": len(result.candidates),
        "label_count": len(result.label_ids),
        "predicted_count": len(predictions),
        "true_count": len(truth),
        "true_point_count": truth_point_count,
        "true_ellipse_count": truth_ellipse_count,
        "count_error": len(predictions) - len(truth),
        "absolute_count_error": abs(len(predictions) - len(truth)),
        "true_positive": true_positive,
        "point_true_positive": point_true_positive,
        "ellipse_true_positive": ellipse_true_positive,
        "point_recall": point_true_positive / max(1, truth_point_count),
        "ellipse_recall": ellipse_true_positive / max(1, truth_ellipse_count),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall),
        "label_region_false_positives": label_region_false_positives,
        "dish_error": _dish_error(result.dish, gold, scale),
        "high_density": len(truth) >= 300,
    }


def evaluate_gold_dataset(
    image_dir: Path,
    annotations_dir: Path,
    pipeline: ColonyCounterPipeline,
    match_tolerance: float = 18.0,
) -> dict[str, Any]:
    annotation_paths = sorted(annotations_dir.glob("*.gold.json"))
    metrics: list[dict[str, Any]] = []
    missing_images: list[str] = []

    for annotation_path in annotation_paths:
        gold = PlateGoldAnnotation.load(annotation_path)
        image_path = image_dir / gold.image
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue
        metrics.append(
            evaluate_gold_image(
                image_path=image_path,
                annotation_path=annotation_path,
                pipeline=pipeline,
                match_tolerance=match_tolerance,
            )
        )

    if not metrics:
        return {
            "images": [],
            "missing_images": missing_images,
            "summary": {"images": 0},
        }

    image_count = len(metrics)
    total_true_positive = sum(item["true_positive"] for item in metrics)
    total_false_positive = sum(item["false_positive"] for item in metrics)
    total_false_negative = sum(item["false_negative"] for item in metrics)
    micro_precision = total_true_positive / max(1, total_true_positive + total_false_positive)
    micro_recall = total_true_positive / max(1, total_true_positive + total_false_negative)

    summary = {
        "images": image_count,
        "missing_images": len(missing_images),
        "true_colonies": sum(item["true_count"] for item in metrics),
        "true_point_colonies": sum(item["true_point_count"] for item in metrics),
        "true_ellipse_colonies": sum(item["true_ellipse_count"] for item in metrics),
        "predicted_colonies": sum(item["predicted_count"] for item in metrics),
        "point_recall": sum(item["point_true_positive"] for item in metrics)
        / max(1, sum(item["true_point_count"] for item in metrics)),
        "ellipse_recall": sum(item["ellipse_true_positive"] for item in metrics)
        / max(1, sum(item["true_ellipse_count"] for item in metrics)),
        "mean_precision": sum(item["precision"] for item in metrics) / image_count,
        "mean_recall": sum(item["recall"] for item in metrics) / image_count,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": 0.0 if micro_precision + micro_recall == 0.0 else (2.0 * micro_precision * micro_recall) / (micro_precision + micro_recall),
        "mean_absolute_count_error": sum(item["absolute_count_error"] for item in metrics) / image_count,
        "total_label_region_false_positives": sum(item["label_region_false_positives"] for item in metrics),
    }
    return {"images": metrics, "missing_images": missing_images, "summary": summary}
