from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DishCircle:
    x: int
    y: int
    radius: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "radius": self.radius}


@dataclass
class Candidate:
    id: int
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: float
    circularity: float
    solidity: float
    equivalent_radius: float
    rim_margin: float
    mean_bgr: tuple[float, float, float]
    mean_lab: tuple[float, float, float]
    local_contrast: float
    edge_strength: float
    contour: Any = field(repr=False, default=None)
    scores: dict[str, float] = field(default_factory=dict)
    predicted_label: bool = False
    predicted_colony: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "center": [float(self.center[0]), float(self.center[1])],
            "bbox": list(self.bbox),
            "area": float(self.area),
            "circularity": float(self.circularity),
            "solidity": float(self.solidity),
            "equivalent_radius": float(self.equivalent_radius),
            "rim_margin": float(self.rim_margin),
            "mean_bgr": [float(v) for v in self.mean_bgr],
            "mean_lab": [float(v) for v in self.mean_lab],
            "local_contrast": float(self.local_contrast),
            "edge_strength": float(self.edge_strength),
            "scores": {key: float(value) for key, value in self.scores.items()},
            "predicted_label": self.predicted_label,
            "predicted_colony": self.predicted_colony,
        }


@dataclass
class PipelineResult:
    dish: DishCircle
    candidates: list[Candidate]
    label_ids: list[int]
    colony_ids: list[int]
    debug_images: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def colony_count(self) -> int:
        return len(self.colony_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dish": self.dish.to_dict(),
            "colony_count": self.colony_count,
            "label_count": len(self.label_ids),
            "candidate_count": len(self.candidates),
            "label_ids": list(self.label_ids),
            "colony_ids": list(self.colony_ids),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

