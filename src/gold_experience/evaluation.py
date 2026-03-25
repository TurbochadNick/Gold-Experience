from __future__ import annotations

import json
from math import hypot
from pathlib import Path
from typing import Any

import cv2

from .pipeline import ColonyCounterPipeline


def _match_predictions(
    predictions: list[tuple[float, float]],
    colonies: list[dict[str, Any]],
    base_tolerance: float = 16.0,
) -> tuple[int, int, int]:
    matched_truth: set[int] = set()
    true_positive = 0

    for pred_x, pred_y in predictions:
        best_index = None
        best_distance = None
        for index, colony in enumerate(colonies):
            if index in matched_truth:
                continue

            tolerance = max(base_tolerance, float(colony.get("radius", 0)) * 1.5)
            distance = hypot(pred_x - float(colony["x"]), pred_y - float(colony["y"]))
            if distance > tolerance:
                continue
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance

        if best_index is not None:
            matched_truth.add(best_index)
            true_positive += 1

    false_positive = len(predictions) - true_positive
    false_negative = len(colonies) - true_positive
    return true_positive, false_positive, false_negative


def evaluate_image(
    image_path: Path,
    metadata_path: Path,
    pipeline: ColonyCounterPipeline,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    result = pipeline.run(image)
    predictions = [
        tuple(candidate.center)
        for candidate in result.candidates
        if candidate.id in set(result.colony_ids)
    ]
    true_positive, false_positive, false_negative = _match_predictions(
        predictions=predictions,
        colonies=metadata.get("colonies", []),
    )

    label_mask_path = metadata_path.with_name(metadata["label_mask"])
    label_mask = cv2.imread(str(label_mask_path), cv2.IMREAD_GRAYSCALE)
    label_false_positives = 0
    if label_mask is not None:
        height, width = label_mask.shape[:2]
        for pred_x, pred_y in predictions:
            x_pos = max(0, min(width - 1, int(round(pred_x))))
            y_pos = max(0, min(height - 1, int(round(pred_y))))
            if label_mask[y_pos, x_pos] > 0:
                label_false_positives += 1

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)

    return {
        "image": image_path.name,
        "predicted_count": len(predictions),
        "true_count": int(metadata.get("colony_count", len(metadata.get("colonies", [])))),
        "count_error": len(predictions) - int(metadata.get("colony_count", len(metadata.get("colonies", [])))),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "label_false_positives": label_false_positives,
    }


def evaluate_dataset(dataset_dir: Path, pipeline: ColonyCounterPipeline) -> dict[str, Any]:
    metadata_paths = sorted(dataset_dir.glob("*.meta.json"))
    metrics = [
        evaluate_image(
            image_path=metadata_path.with_name(json.loads(metadata_path.read_text(encoding="utf-8"))["image"]),
            metadata_path=metadata_path,
            pipeline=pipeline,
        )
        for metadata_path in metadata_paths
    ]

    if not metrics:
        return {"images": [], "summary": {"images": 0}}

    image_count = len(metrics)
    summary = {
        "images": image_count,
        "mean_precision": sum(item["precision"] for item in metrics) / image_count,
        "mean_recall": sum(item["recall"] for item in metrics) / image_count,
        "mean_absolute_count_error": sum(abs(item["count_error"]) for item in metrics) / image_count,
        "total_label_false_positives": sum(item["label_false_positives"] for item in metrics),
    }
    return {"images": metrics, "summary": summary}

