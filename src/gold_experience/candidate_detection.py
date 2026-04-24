from __future__ import annotations

from dataclasses import dataclass
from math import hypot, pi, sqrt

import cv2
import numpy as np

from .models import Candidate, DishCircle


@dataclass
class Proposal:
    center: tuple[float, float]
    radius: float
    feature: np.ndarray
    source_rank: float


def _normalize_feature(feature: np.ndarray) -> np.ndarray:
    if float(np.max(feature)) <= float(np.min(feature)):
        return np.zeros_like(feature, dtype=np.uint8)
    return cv2.normalize(feature, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _build_blob_detector(min_area: float, max_area: float, min_distance: float) -> cv2.SimpleBlobDetector:
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 8
    params.maxThreshold = 220
    params.thresholdStep = 8
    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)
    params.filterByColor = True
    params.blobColor = 255
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minDistBetweenBlobs = float(min_distance)
    params.minRepeatability = 1
    return cv2.SimpleBlobDetector_create(params)


def _detect_blob_proposals(
    feature: np.ndarray,
    dish_mask: np.ndarray,
    min_area: float,
    max_area: float,
    min_distance: float,
    source_rank: float,
) -> list[Proposal]:
    masked_feature = cv2.bitwise_and(feature, dish_mask)
    detector = _build_blob_detector(min_area=min_area, max_area=max_area, min_distance=min_distance)
    keypoints = detector.detect(masked_feature)
    proposals: list[Proposal] = []
    for keypoint in keypoints:
        proposals.append(
            Proposal(
                center=(float(keypoint.pt[0]), float(keypoint.pt[1])),
                radius=max(3.0, float(keypoint.size) / 2.0),
                feature=feature,
                source_rank=source_rank,
            )
        )
    return proposals


def _contour_from_mask(region_mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _region_from_proposal(proposal: Proposal, dish_mask: np.ndarray) -> np.ndarray:
    height, width = proposal.feature.shape[:2]
    x_center = int(round(proposal.center[0]))
    y_center = int(round(proposal.center[1]))
    patch_radius = int(max(8.0, proposal.radius * 2.5))

    x0 = max(0, x_center - patch_radius)
    y0 = max(0, y_center - patch_radius)
    x1 = min(width, x_center + patch_radius + 1)
    y1 = min(height, y_center + patch_radius + 1)

    patch = proposal.feature[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros((height, width), dtype=np.uint8)

    local_center_x = x_center - x0
    local_center_y = y_center - y0
    seed_value = float(patch[local_center_y, local_center_x])
    threshold_value = max(
        12.0,
        seed_value * 0.55,
        float(np.mean(patch)) + 0.30 * float(np.std(patch)),
    )
    local_binary = np.zeros_like(patch, dtype=np.uint8)
    local_binary[patch >= threshold_value] = 255

    limit_mask = np.zeros_like(local_binary, dtype=np.uint8)
    cv2.circle(
        limit_mask,
        (local_center_x, local_center_y),
        int(max(proposal.radius * 1.8, proposal.radius + 4.0)),
        255,
        -1,
        lineType=cv2.LINE_AA,
    )
    local_binary = cv2.bitwise_and(local_binary, limit_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    local_binary = cv2.morphologyEx(local_binary, cv2.MORPH_OPEN, kernel, iterations=1)
    local_binary = cv2.morphologyEx(local_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    labels_count, labels = cv2.connectedComponents(local_binary)
    component_label = labels[local_center_y, local_center_x]
    region_mask = np.zeros((height, width), dtype=np.uint8)

    if labels_count > 1 and component_label > 0:
        component = np.zeros_like(local_binary, dtype=np.uint8)
        component[labels == component_label] = 255
        region_mask[y0:y1, x0:x1] = component
    else:
        cv2.circle(
            region_mask,
            (x_center, y_center),
            int(round(proposal.radius * 1.15)),
            255,
            -1,
            lineType=cv2.LINE_AA,
        )

    return cv2.bitwise_and(region_mask, dish_mask)


def _candidate_from_mask(
    region_mask: np.ndarray,
    image: np.ndarray,
    lab: np.ndarray,
    combined_feature: np.ndarray,
    abs_laplacian: np.ndarray,
    dish: DishCircle,
    proposal_rank: float,
) -> Candidate | None:
    contour = _contour_from_mask(region_mask)
    if contour is None:
        return None

    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return None

    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0.0:
        return None

    moments = cv2.moments(contour)
    if float(moments["m00"]) == 0.0:
        return None

    center_x = float(moments["m10"] / moments["m00"])
    center_y = float(moments["m01"] / moments["m00"])
    x_pos, y_pos, box_width, box_height = cv2.boundingRect(contour)
    circularity = 4.0 * pi * area / (perimeter * perimeter)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0.0 else 0.0
    equivalent_radius = sqrt(area / pi)
    center_distance = hypot(center_x - dish.x, center_y - dish.y)
    rim_margin = float(dish.radius - center_distance - equivalent_radius)

    mean_bgr = cv2.mean(image, mask=region_mask)[:3]
    mean_lab = cv2.mean(lab, mask=region_mask)[:3]
    local_contrast = float(cv2.mean(combined_feature, mask=region_mask)[0])
    edge_strength = float(cv2.mean(abs_laplacian, mask=region_mask)[0])

    candidate = Candidate(
        id=-1,
        center=(center_x, center_y),
        bbox=(x_pos, y_pos, box_width, box_height),
        area=area,
        circularity=float(circularity),
        solidity=float(solidity),
        equivalent_radius=float(equivalent_radius),
        rim_margin=rim_margin,
        mean_bgr=tuple(float(v) for v in mean_bgr),
        mean_lab=tuple(float(v) for v in mean_lab),
        local_contrast=local_contrast,
        edge_strength=edge_strength,
        contour=contour,
    )
    candidate.scores["proposal_rank"] = float(proposal_rank)
    return candidate


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    if not candidates:
        return []

    def quality(candidate: Candidate) -> float:
        return (
            candidate.local_contrast
            + 18.0 * candidate.solidity
            + 12.0 * candidate.circularity
            + 0.012 * candidate.area
            + 6.0 * candidate.scores.get("proposal_rank", 0.0)
        )

    def iou(left: Candidate, right: Candidate) -> float:
        left_x, left_y, left_w, left_h = left.bbox
        right_x, right_y, right_w, right_h = right.bbox
        x0 = max(left_x, right_x)
        y0 = max(left_y, right_y)
        x1 = min(left_x + left_w, right_x + right_w)
        y1 = min(left_y + left_h, right_y + right_h)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        intersection = float((x1 - x0) * (y1 - y0))
        union = float(left_w * left_h + right_w * right_h) - intersection
        return 0.0 if union <= 0.0 else intersection / union

    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=quality, reverse=True):
        duplicate = False
        for existing in kept:
            center_distance = hypot(candidate.center[0] - existing.center[0], candidate.center[1] - existing.center[1])
            radius_gate = 0.45 * (candidate.equivalent_radius + existing.equivalent_radius)
            if center_distance <= radius_gate or iou(candidate, existing) >= 0.35:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    for index, candidate in enumerate(kept):
        candidate.id = index
    return kept


def detect_candidates(
    image: np.ndarray,
    dish: DishCircle,
    dish_mask: np.ndarray,
    baseline: dict[str, float] | None = None,
) -> tuple[list[Candidate], dict[str, np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)

    dark_small = _normalize_feature(
        cv2.subtract(cv2.GaussianBlur(gray_blur, (0, 0), sigmaX=9, sigmaY=9), gray_blur)
    )
    dark_large = _normalize_feature(
        cv2.subtract(cv2.GaussianBlur(gray_blur, (0, 0), sigmaX=35, sigmaY=35), gray_blur)
    )

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1].astype(np.float32)
    b_channel = lab[:, :, 2].astype(np.float32)
    warm_raw = (
        0.65 * np.clip(b_channel - 128.0, 0.0, None)
        + 0.35 * np.clip(a_channel - 128.0, 0.0, None)
        - 0.30 * np.clip(118.0 - a_channel, 0.0, None)
    )
    warm_raw = np.clip(warm_raw, 0.0, None)
    warm_small = _normalize_feature(warm_raw - cv2.GaussianBlur(warm_raw, (0, 0), sigmaX=9, sigmaY=9))
    warm_large = _normalize_feature(warm_raw - cv2.GaussianBlur(warm_raw, (0, 0), sigmaX=21, sigmaY=21))

    small_feature = cv2.max(dark_small, warm_small)
    large_feature = cv2.max(dark_large, warm_large)
    combined_feature = cv2.max(small_feature, large_feature)
    masked_combined = cv2.bitwise_and(combined_feature, dish_mask)
    _, threshold = cv2.threshold(masked_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    laplacian = cv2.Laplacian(gray_blur, cv2.CV_32F)
    abs_laplacian = np.abs(laplacian)
    dish_area = pi * dish.radius * dish.radius
    agar_l = 210.0 if baseline is None else float(baseline.get("L", 210.0))
    agar_l_std = 18.0 if baseline is None else float(baseline.get("L_std", 18.0))
    contrast_floor = agar_l_std * 3.2

    proposals: list[Proposal] = []
    proposals.extend(
        _detect_blob_proposals(
            feature=cv2.bitwise_and(small_feature, dish_mask),
            dish_mask=dish_mask,
            min_area=max(8.0, dish_area * 0.000008),
            max_area=max(400.0, dish_area * 0.003),
            min_distance=4.0,
            source_rank=2.5,
        )
    )
    proposals.extend(
        _detect_blob_proposals(
            feature=cv2.bitwise_and(large_feature, dish_mask),
            dish_mask=dish_mask,
            min_area=max(18.0, dish_area * 0.000015),
            max_area=max(1200.0, dish_area * 0.018),
            min_distance=8.0,
            source_rank=2.0,
        )
    )
    proposals.extend(
        _detect_blob_proposals(
            feature=cv2.bitwise_and(dark_small, dish_mask),
            dish_mask=dish_mask,
            min_area=max(6.0, dish_area * 0.000006),
            max_area=max(250.0, dish_area * 0.0018),
            min_distance=3.0,
            source_rank=1.5,
        )
    )

    raw_candidates: list[Candidate] = []
    for proposal in proposals:
        region_mask = _region_from_proposal(proposal=proposal, dish_mask=dish_mask)
        candidate = _candidate_from_mask(
            region_mask=region_mask,
            image=image,
            lab=lab,
            combined_feature=combined_feature,
            abs_laplacian=abs_laplacian,
            dish=dish,
            proposal_rank=proposal.source_rank,
        )
        if candidate is None:
            continue
        if candidate.area < max(6.0, dish_area * 0.000006):
            continue
        if candidate.area > dish_area * 0.045:
            continue
        lightness, a_value, b_value = candidate.mean_lab
        chroma = abs(a_value - 128.0) + abs(b_value - 128.0)
        if lightness > agar_l + 36.0 and chroma < 24.0:
            continue
        if candidate.local_contrast < contrast_floor and chroma < 28.0:
            continue
        if candidate.area < max(16.0, dish_area * 0.000015) and candidate.local_contrast < contrast_floor * 1.3:
            continue
        raw_candidates.append(candidate)

    candidates = _dedupe_candidates(raw_candidates)

    candidate_mask = np.zeros_like(gray)
    for candidate in candidates:
        cv2.drawContours(candidate_mask, [candidate.contour], -1, 255, -1, lineType=cv2.LINE_AA)

    debug = {
        "dark_feature": dark_large,
        "warm_feature": warm_large,
        "candidate_threshold": threshold,
        "candidate_mask": candidate_mask,
    }
    return candidates, debug
