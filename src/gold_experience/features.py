from __future__ import annotations


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def lab_warmth(mean_lab: tuple[float, float, float]) -> float:
    _, a_value, b_value = mean_lab
    b_score = clamp((b_value - 145.0) / 35.0)
    a_score = clamp((a_value - 122.0) / 18.0)
    green_penalty = clamp((118.0 - a_value) / 28.0)
    return clamp(0.6 * b_score + 0.4 * a_score - 0.35 * green_penalty)


def lab_darkness(mean_lab: tuple[float, float, float]) -> float:
    lightness, _, _ = mean_lab
    return clamp((150.0 - lightness) / 85.0)


def lab_chroma(mean_lab: tuple[float, float, float]) -> float:
    _, a_value, b_value = mean_lab
    return abs(a_value - 128.0) + abs(b_value - 128.0)


def lab_neutrality(mean_lab: tuple[float, float, float]) -> float:
    return 1.0 - clamp(lab_chroma(mean_lab) / 95.0)


def angle_to_grid_score(angle_degrees: float) -> float:
    wrapped = angle_degrees % 45.0
    delta = min(wrapped, 45.0 - wrapped)
    return 1.0 - clamp(delta / 15.0)
