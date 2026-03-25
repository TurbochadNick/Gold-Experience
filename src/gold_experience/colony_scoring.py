from __future__ import annotations

from math import pi

from .features import clamp, lab_warmth
from .models import Candidate, DishCircle


def classify_colonies(
    candidates: list[Candidate],
    label_ids: list[int],
    dish: DishCircle,
) -> list[int]:
    label_id_set = set(label_ids)
    colony_ids: list[int] = []

    dish_area = pi * dish.radius * dish.radius
    min_area = max(14.0, dish_area * 0.000015)
    max_area = dish_area * 0.03

    for candidate in candidates:
        if candidate.id in label_id_set:
            candidate.predicted_colony = False
            continue

        size_score = 1.0 if min_area <= candidate.area <= max_area else 0.0
        shape_score = clamp((candidate.circularity - 0.25) / 0.45)
        solidity_score = clamp((candidate.solidity - 0.75) / 0.2)
        contrast_score = clamp(candidate.local_contrast / 145.0)
        warmth_score = lab_warmth(candidate.mean_lab)
        large_blob_score = clamp((candidate.equivalent_radius - 8.0) / 12.0)
        rim_penalty = 0.5 if candidate.rim_margin <= max(4.0, 0.35 * candidate.equivalent_radius) else 0.0
        text_penalty = 0.55 * candidate.scores.get("label_score", 0.0)

        colony_score = (
            0.25 * size_score
            + 0.20 * shape_score
            + 0.15 * solidity_score
            + 0.20 * contrast_score
            + 0.20 * max(warmth_score, large_blob_score)
            - rim_penalty
            - text_penalty
        )

        candidate.scores["colony_score"] = float(colony_score)
        candidate.predicted_colony = colony_score >= 0.45
        if candidate.predicted_colony:
            colony_ids.append(candidate.id)

    return sorted(colony_ids)
