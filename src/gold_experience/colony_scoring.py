from __future__ import annotations

from math import pi

from .features import clamp, lab_darkness, lab_warmth, relative_contrast_score
from .models import Candidate, DishCircle


def classify_colonies(
    candidates: list[Candidate],
    label_ids: list[int],
    dish: DishCircle,
    agar_baseline: dict[str, float] | None = None,
) -> list[int]:
    label_id_set = set(label_ids)
    colony_ids: list[int] = []
    agar_l = agar_baseline["L"] if agar_baseline else 210.0
    agar_a = agar_baseline["a"] if agar_baseline else 128.0
    agar_b = agar_baseline["b"] if agar_baseline else 128.0
    agar_l_std = agar_baseline["L_std"] if agar_baseline else 18.0

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
        contrast_score = relative_contrast_score(candidate.local_contrast, agar_l_std)
        warmth_score = lab_warmth(candidate.mean_lab, agar_a=agar_a, agar_b=agar_b)
        darkness_score = lab_darkness(candidate.mean_lab, agar_l=agar_l)
        large_blob_score = clamp((candidate.equivalent_radius - 8.0) / 12.0)
        rim_penalty = 0.5 if candidate.rim_margin <= max(4.0, 0.35 * candidate.equivalent_radius) else 0.0
        text_penalty = 0.55 * candidate.scores.get("label_score", 0.0)

        colony_score = (
            0.25 * size_score
            + 0.18 * shape_score
            + 0.12 * solidity_score
            + 0.18 * contrast_score
            + 0.15 * max(warmth_score, large_blob_score)
            + 0.12 * darkness_score
            - rim_penalty
            - text_penalty
        )

        candidate.scores["colony_score"] = float(colony_score)
        candidate.scores["darkness_score"] = float(darkness_score)
        candidate.scores["contrast_score"] = float(contrast_score)
        candidate.predicted_colony = colony_score >= 0.45
        if candidate.predicted_colony:
            colony_ids.append(candidate.id)

    return sorted(colony_ids)
