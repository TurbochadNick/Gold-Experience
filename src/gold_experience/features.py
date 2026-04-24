from __future__ import annotations


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def lab_warmth(
    mean_lab: tuple[float, float, float],
    agar_a: float = 128.0,
    agar_b: float = 128.0,
) -> float:
    _, a_value, b_value = mean_lab
    b_score = clamp((b_value - (agar_b + 17.0)) / 35.0)
    a_score = clamp((a_value - (agar_a - 6.0)) / 18.0)
    green_penalty = clamp(((agar_a - 10.0) - a_value) / 28.0)
    return clamp(0.6 * b_score + 0.4 * a_score - 0.35 * green_penalty)


def lab_darkness(
    mean_lab: tuple[float, float, float],
    agar_l: float = 210.0,
) -> float:
    lightness, _, _ = mean_lab
    delta = agar_l - lightness
    return clamp(delta / 40.0)


def lab_chroma(mean_lab: tuple[float, float, float]) -> float:
    _, a_value, b_value = mean_lab
    return abs(a_value - 128.0) + abs(b_value - 128.0)


def lab_neutrality(mean_lab: tuple[float, float, float]) -> float:
    return 1.0 - clamp(lab_chroma(mean_lab) / 95.0)


def angle_to_grid_score(angle_degrees: float) -> float:
    wrapped = angle_degrees % 45.0
    delta = min(wrapped, 45.0 - wrapped)
    return 1.0 - clamp(delta / 15.0)


def relative_contrast_score(local_contrast: float, agar_l_std: float) -> float:
    reference = max(18.0, agar_l_std * 4.5)
    return clamp(local_contrast / reference)
