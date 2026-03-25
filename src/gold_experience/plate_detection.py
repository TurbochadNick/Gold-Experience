from __future__ import annotations

import cv2
import numpy as np

from .models import DishCircle


def _default_dish(height: int, width: int) -> DishCircle:
    min_side = min(height, width)
    return DishCircle(x=width // 2, y=height // 2, radius=int(min_side * 0.40))


def _contour_circle_fallback(
    blurred: np.ndarray,
    min_radius: int,
    max_radius: int,
) -> DishCircle | None:
    height, width = blurred.shape[:2]
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
        iterations=2,
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_circle: DishCircle | None = None
    best_score = None
    image_area = float(height * width)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.10:
            continue

        x_pos, y_pos, box_width, box_height = cv2.boundingRect(contour)
        touches_border = (
            x_pos <= 2
            or y_pos <= 2
            or x_pos + box_width >= width - 2
            or y_pos + box_height >= height - 2
        )
        if touches_border:
            continue

        (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
        if radius < min_radius or radius > max_radius:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter <= 0.0 else (4.0 * np.pi * area) / (perimeter * perimeter)
        center_offset = np.hypot(circle_x - (width / 2.0), circle_y - (height / 2.0))
        score = area + circularity * 1000.0 - center_offset * 4.0
        if best_score is None or score > best_score:
            best_score = score
            best_circle = DishCircle(
                x=int(round(circle_x)),
                y=int(round(circle_y)),
                radius=int(round(radius)),
            )

    return best_circle


def detect_plate(image: np.ndarray) -> tuple[DishCircle, np.ndarray, dict[str, np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)
    h, w = gray.shape[:2]
    min_side = min(h, w)
    min_radius = int(min_side * 0.32)
    max_radius = int(min_side * 0.48)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_side // 2,
        param1=120,
        param2=26,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None and len(circles[0]) > 0:
        x_pos, y_pos, radius = max(
            circles[0],
            key=lambda row: row[2] - 0.35 * np.hypot(row[0] - (w / 2.0), row[1] - (h / 2.0)),
        )
        dish = DishCircle(x=int(round(x_pos)), y=int(round(y_pos)), radius=int(round(radius)))
    else:
        dish = _contour_circle_fallback(
            blurred=blurred,
            min_radius=min_radius,
            max_radius=max_radius,
        )
        if dish is None:
            dish = _default_dish(height=h, width=w)

    mask = np.zeros_like(gray)
    cv2.circle(mask, (dish.x, dish.y), int(dish.radius * 0.94), 255, -1, lineType=cv2.LINE_AA)
    debug = {"plate_gray": gray, "plate_mask": mask}
    return dish, mask, debug
