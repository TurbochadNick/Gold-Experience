from __future__ import annotations

import numpy as np

from .candidate_detection import detect_candidates
from .colony_scoring import classify_colonies
from .illumination import (
    apply_clahe_to_lightness,
    estimate_agar_baseline,
    normalize_to_agar_baseline,
)
from .label_filter import classify_labels
from .models import PipelineResult
from .plate_detection import detect_plate


class ColonyCounterPipeline:
    def run(self, image: np.ndarray) -> PipelineResult:
        clahe_image = apply_clahe_to_lightness(image)
        dish, dish_mask, plate_debug = detect_plate(clahe_image)
        measured_baseline = estimate_agar_baseline(image=clahe_image, dish=dish, dish_mask=dish_mask)
        normalized_image = normalize_to_agar_baseline(image=clahe_image, baseline=measured_baseline)
        baseline = {
            "L": 210.0,
            "a": 128.0,
            "b": 128.0,
            "L_std": measured_baseline.get("L_std", 18.0),
            "source_L": measured_baseline.get("L", 210.0),
            "source_a": measured_baseline.get("a", 128.0),
            "source_b": measured_baseline.get("b", 128.0),
        }

        candidates, candidate_debug = detect_candidates(
            image=normalized_image,
            dish=dish,
            dish_mask=dish_mask,
            baseline=baseline,
        )
        label_ids = classify_labels(candidates)
        colony_ids = classify_colonies(
            candidates=candidates,
            label_ids=label_ids,
            dish=dish,
            agar_baseline=baseline,
        )
        debug_images = dict(plate_debug)
        debug_images["plate_clahe"] = clahe_image
        debug_images["plate_normalized"] = normalized_image
        debug_images.update(candidate_debug)
        return PipelineResult(
            dish=dish,
            candidates=candidates,
            label_ids=label_ids,
            colony_ids=colony_ids,
            agar_baseline=baseline,
            debug_images=debug_images,
        )
