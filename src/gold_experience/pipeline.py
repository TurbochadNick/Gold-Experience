from __future__ import annotations

import numpy as np

from .candidate_detection import detect_candidates
from .colony_scoring import classify_colonies
from .label_filter import classify_labels
from .models import PipelineResult
from .plate_detection import detect_plate


class ColonyCounterPipeline:
    def run(self, image: np.ndarray) -> PipelineResult:
        dish, dish_mask, plate_debug = detect_plate(image)
        candidates, candidate_debug = detect_candidates(image=image, dish=dish, dish_mask=dish_mask)
        label_ids = classify_labels(candidates)
        colony_ids = classify_colonies(candidates=candidates, label_ids=label_ids, dish=dish)
        debug_images = dict(plate_debug)
        debug_images.update(candidate_debug)
        return PipelineResult(
            dish=dish,
            candidates=candidates,
            label_ids=label_ids,
            colony_ids=colony_ids,
            debug_images=debug_images,
        )

