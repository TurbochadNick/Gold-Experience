from __future__ import annotations

from typing import Any

from .features import clamp
from .models import Candidate, PipelineResult


def _size_bucket(radius: float) -> str:
    if radius < 5.0:
        return "small"
    if radius < 9.0:
        return "medium"
    return "large"


def _candidate_payload(candidate: Candidate, kind: str) -> dict[str, Any]:
    colony_score = clamp(candidate.scores.get("colony_score", 0.0))
    label_score = clamp(candidate.scores.get("label_score", 0.0))
    radius = max(4.0, float(candidate.equivalent_radius))
    return {
        "id": f"{kind}-{candidate.id}",
        "candidate_id": int(candidate.id),
        "kind": kind,
        "x": float(candidate.center[0]),
        "y": float(candidate.center[1]),
        "r": radius,
        "bbox": [int(value) for value in candidate.bbox],
        "area": float(candidate.area),
        "size": _size_bucket(radius),
        "conf": float(colony_score if kind == "colony" else label_score),
        "colony_score": float(colony_score),
        "label_score": float(label_score),
        "rim_margin": float(candidate.rim_margin),
        "circularity": float(candidate.circularity),
        "solidity": float(candidate.solidity),
        "local_contrast": float(candidate.local_contrast),
        "edge_strength": float(candidate.edge_strength),
    }


def build_frontend_payload(
    result: PipelineResult,
    image_shape: tuple[int, int, int] | tuple[int, int],
    filename: str | None = None,
) -> dict[str, Any]:
    height = int(image_shape[0])
    width = int(image_shape[1])
    label_id_set = set(result.label_ids)
    colony_id_set = set(result.colony_ids)

    colonies: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for candidate in result.candidates:
        if candidate.id in colony_id_set:
            colonies.append(_candidate_payload(candidate, kind="colony"))
        elif candidate.id in label_id_set:
            labels.append(_candidate_payload(candidate, kind="label"))
        else:
            rejected.append(_candidate_payload(candidate, kind="rejected"))

    colonies.sort(key=lambda item: (-item["conf"], item["candidate_id"]))
    labels.sort(key=lambda item: (-item["conf"], item["candidate_id"]))
    rejected.sort(key=lambda item: (-item["conf"], item["candidate_id"]))

    average_confidence = (
        sum(item["conf"] for item in colonies) / len(colonies) if colonies else 0.0
    )

    return {
        "engine": "gold-experience-v1",
        "image": {
            "filename": filename or "",
            "width": width,
            "height": height,
        },
        "dish": result.dish.to_dict(),
        "agar_baseline": None if result.agar_baseline is None else {key: float(value) for key, value in result.agar_baseline.items()},
        "summary": {
            "candidate_count": len(result.candidates),
            "colony_count": len(colonies),
            "label_count": len(labels),
            "rejected_count": len(rejected),
            "average_confidence": float(average_confidence),
        },
        "pipeline_steps": [
            {
                "key": "dish_detection",
                "label": "Dish Detection",
                "status": "done",
                "detail": f"r={result.dish.radius}px",
            },
            {
                "key": "candidate_detection",
                "label": "Candidate Detection",
                "status": "done",
                "detail": f"{len(result.candidates)} proposals",
            },
            {
                "key": "label_filter",
                "label": "Label Filter",
                "status": "done",
                "detail": f"{len(labels)} rejected as labels",
            },
            {
                "key": "colony_scoring",
                "label": "Colony Scoring",
                "status": "done",
                "detail": f"{len(colonies)} colonies kept",
            },
            {
                "key": "manual_review",
                "label": "Manual Review",
                "status": "done",
                "detail": "Ready",
            },
        ],
        "colonies": colonies,
        "labels": labels,
        "rejected": rejected,
    }
