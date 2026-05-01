from __future__ import annotations

from collections import defaultdict, deque
from math import atan2, degrees, hypot

import numpy as np

from .features import angle_to_grid_score, clamp, lab_darkness, lab_neutrality, lab_warmth
from .models import Candidate


def classify_labels(candidates: list[Candidate]) -> list[int]:
    if not candidates:
        return []

    neighbors: dict[int, list[dict[str, float]]] = {candidate.id: [] for candidate in candidates}
    adjacency: dict[int, set[int]] = defaultdict(set)
    cluster_adjacency: dict[int, set[int]] = defaultdict(set)

    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            distance = hypot(left.center[0] - right.center[0], left.center[1] - right.center[1])
            if distance <= 0.0:
                continue

            size_ratio = max(left.area, right.area) / max(1.0, min(left.area, right.area))
            if size_ratio > 3.0:
                continue

            max_neighbor_distance = max(30.0, 6.0 * max(left.equivalent_radius, right.equivalent_radius))
            if distance > max_neighbor_distance:
                continue

            angle = abs(degrees(atan2(right.center[1] - left.center[1], right.center[0] - left.center[0])))
            grid_score = angle_to_grid_score(angle)
            record = {"distance": distance, "size_ratio": size_ratio, "grid_score": grid_score}
            neighbors[left.id].append(record)
            neighbors[right.id].append(record)

            if grid_score >= 0.45:
                adjacency[left.id].add(right.id)
                adjacency[right.id].add(left.id)
            if distance <= max(24.0, 4.5 * max(left.equivalent_radius, right.equivalent_radius)) and size_ratio <= 2.0:
                cluster_adjacency[left.id].add(right.id)
                cluster_adjacency[right.id].add(left.id)

    for candidate in candidates:
        nearby = neighbors[candidate.id]
        nearby_count = len(nearby)
        aligned_neighbors = sum(1 for item in nearby if item["grid_score"] >= 0.65)
        grid_mean = sum(item["grid_score"] for item in nearby) / len(nearby) if nearby else 0.0
        small_score = clamp((13.0 - candidate.equivalent_radius) / 9.0)
        dark_score = lab_darkness(candidate.mean_lab)
        neutral_score = lab_neutrality(candidate.mean_lab)
        warm_score = lab_warmth(candidate.mean_lab)
        warmth_veto = clamp(warm_score / 0.35)
        label_score = (
            0.25 * small_score
            + 0.25 * ((dark_score + neutral_score) / 2.0)
            + 0.20 * clamp(aligned_neighbors / 2.0)
            + 0.15 * grid_mean
            + 0.15 * clamp(nearby_count / 4.0)
            - 0.45 * warmth_veto
        )
        candidate.scores.update(
            {
                "nearby_count": float(nearby_count),
                "aligned_neighbors": float(aligned_neighbors),
                "grid_mean": float(grid_mean),
                "label_score": float(label_score),
                "warmth": float(warm_score),
                "warmth_veto": float(warmth_veto),
                "darkness": float(dark_score),
                "neutrality": float(neutral_score),
            }
        )

    label_ids: set[int] = set()
    visited: set[int] = set()
    by_id = {candidate.id: candidate for candidate in candidates}

    for candidate in candidates:
        if candidate.id in visited:
            continue

        queue = deque([candidate.id])
        component: list[int] = []
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor_id in cluster_adjacency[current]:
                if neighbor_id not in visited:
                    queue.append(neighbor_id)

        if len(component) < 5:
            continue

        component_candidates = [by_id[candidate_id] for candidate_id in component]
        mean_label_score = sum(item.scores["label_score"] for item in component_candidates) / len(component_candidates)
        mean_warmth = sum(item.scores["warmth"] for item in component_candidates) / len(component_candidates)
        mean_radius = sum(item.equivalent_radius for item in component_candidates) / len(component_candidates)
        mean_alignment = sum(item.scores["aligned_neighbors"] for item in component_candidates) / len(component_candidates)
        radius_values = np.array([item.equivalent_radius for item in component_candidates], dtype=np.float32)
        radius_cv = float(radius_values.std() / max(radius_values.mean(), 1.0))

        if (
            mean_label_score >= 0.38
            and mean_warmth <= 0.22
            and mean_radius <= 7.5
            and radius_cv <= 0.35
            and mean_alignment >= 0.2
        ):
            label_ids.update(component)

    for candidate in candidates:
        if candidate.id in label_ids:
            candidate.predicted_label = True
            continue

        if (
            candidate.scores["label_score"] >= 0.52
            and candidate.scores["aligned_neighbors"] >= 2.0
            and candidate.scores["grid_mean"] >= 0.55
            and candidate.scores["nearby_count"] >= 3.0
            and candidate.scores["warmth"] < 0.30
        ):
            label_ids.add(candidate.id)
            candidate.predicted_label = True

    for candidate in candidates:
        candidate.predicted_label = candidate.id in label_ids

    return sorted(label_ids)
