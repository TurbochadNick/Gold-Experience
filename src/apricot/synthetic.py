from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .plate_schema import (
    DEFAULT_SYNTHETIC_SCHEMA,
    PLATE_SCHEMA_LABELS,
    PLATE_SCHEMA_REGISTRY,
    SCHEMA_CLEAN_DOTS,
    SCHEMA_MERGED_SNOWMAN,
    SCHEMA_MIXED_PLATE,
    SCHEMA_STREAK_LINES,
    SCHEMA_UNKNOWN,
    plate_schema_registry_payload,
)

LOGGER = logging.getLogger(__name__)

GENERATOR_VERSION = "apricot-synthetic-v3.2"
CLASS_NAME = "yeast_colony"
CLASS_NAMES = [CLASS_NAME]
SYNTHETIC_SCHEMAS = (SCHEMA_CLEAN_DOTS, SCHEMA_MERGED_SNOWMAN)
MERGED_CLUSTER_SIZE_RANGE = (2, 4)

DEFAULT_IMAGE_SIZE = 2000
DEFAULT_DISH_RADIUS_RATIO = 0.45
DEFAULT_COLLISION_MARGIN = 5
BLACK_BACKGROUND_HEX = "#000000"
DEFAULT_AGAR_HEX = "#3A3420"
ALT_AGAR_HEX = "#544C2E"
DISH_OUTLINE_HEX = "#FFFFFF"
COLONY_CENTER_HEX = "#E4DCC6"
COLONY_EDGE_HEX = "#9C744A"

SIZE_RANGES: dict[str, tuple[int, int]] = {
    "small": (15, 25),
    "medium": (25, 40),
    "large": (40, 60),
    "mixed": (15, 60),
}

DEFAULT_COLONY_RANGES: dict[str, tuple[int, int]] = {
    "small": (60, 120),
    "medium": (50, 100),
    "large": (35, 75),
    "mixed": (50, 120),
}

STANDARD_SPLITS = ("train_standard", "val_standard")
STRESS_SPLITS = (
    "test_lighting_shift",
    "test_agar_color_shift",
    "test_blur_compression",
    "test_density_extremes",
    "test_size_extremes",
    "test_plate_position_shift",
    "test_artifact_noise",
)
SYNTHETIC_SUITE_SPLITS = STANDARD_SPLITS + STRESS_SPLITS


def _clone_ranges(ranges: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    return {key: (int(value[0]), int(value[1])) for key, value in ranges.items()}


@dataclass(frozen=True)
class SpeciesProfile:
    id: str
    display_name: str
    aliases: tuple[str, ...] = ()
    colony_radius_ranges: dict[str, tuple[int, int]] = field(default_factory=lambda: _clone_ranges(SIZE_RANGES))
    density_ranges: dict[str, tuple[int, int]] = field(default_factory=lambda: _clone_ranges(DEFAULT_COLONY_RANGES))
    colony_center_hex: str = COLONY_CENTER_HEX
    colony_edge_hex: str = COLONY_EDGE_HEX


@dataclass(frozen=True)
class MediaProfile:
    id: str
    display_name: str
    aliases: tuple[str, ...] = ()
    agar_hex: str = DEFAULT_AGAR_HEX
    use_cases: tuple[str, ...] = ()
    selection_markers: tuple[str, ...] = ()
    base_profile: str | None = None


@dataclass(frozen=True)
class SyntheticDomainConfig:
    species: str = "generic_yeast"
    medium: str = "generic_dark_agar"
    colony_radius_ranges: dict[str, tuple[int, int]] = field(default_factory=lambda: _clone_ranges(SIZE_RANGES))
    colony_center_hex: str = COLONY_CENTER_HEX
    colony_edge_hex: str = COLONY_EDGE_HEX
    agar_hex: str = DEFAULT_AGAR_HEX
    density_ranges: dict[str, tuple[int, int]] = field(default_factory=lambda: _clone_ranges(DEFAULT_COLONY_RANGES))
    image_size: int = DEFAULT_IMAGE_SIZE
    dish_radius: int | None = None
    random_seed: int | None = None
    collision_margin: int = DEFAULT_COLLISION_MARGIN
    selection_markers: tuple[str, ...] = ()
    dish_center_offset: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class ImageStressConfig:
    brightness_delta: float = 0.0
    contrast: float = 1.0
    gradient_strength: float = 0.0
    vignette_strength: float = 0.0
    blur_sigma: float = 0.0
    jpeg_quality: int = 94
    artifact_noise: bool = False
    artifact_count_range: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class SyntheticSplitSpec:
    name: str
    tier: str
    plates: int
    size_mode: str = "starter"
    species: str = "generic_yeast"
    medium: str = "generic_dark_agar"
    agar_hex: str | None = None
    colony_center_hex: str | None = None
    colony_edge_hex: str | None = None
    colony_radius_ranges: dict[str, tuple[int, int]] | None = None
    density_ranges: dict[str, tuple[int, int]] | None = None
    dish_offset_fraction_range: tuple[float, float] = (0.0, 0.0)
    image_stress: ImageStressConfig = field(default_factory=ImageStressConfig)
    notes: str = ""


SPECIES_PROFILES: dict[str, SpeciesProfile] = {
    "generic_yeast": SpeciesProfile(
        id="generic_yeast",
        display_name="Generic yeast",
    ),
    "s_cerevisiae": SpeciesProfile(
        id="s_cerevisiae",
        display_name="Saccharomyces cerevisiae",
        aliases=("saccharomyces_cerevisiae", "saccharomyces cerevisiae", "scerevisiae"),
        colony_radius_ranges={
            "small": (12, 22),
            "medium": (22, 36),
            "large": (36, 54),
            "mixed": (12, 54),
        },
        density_ranges={
            "small": (70, 130),
            "medium": (55, 105),
            "large": (35, 80),
            "mixed": (50, 125),
        },
    ),
    "p_pastoris": SpeciesProfile(
        id="p_pastoris",
        display_name="Pichia pastoris / Komagataella phaffii",
        aliases=("pichia_pastoris", "pichia pastoris", "komagataella_phaffii", "komagataella phaffii"),
        colony_radius_ranges={
            "small": (10, 20),
            "medium": (20, 34),
            "large": (34, 50),
            "mixed": (10, 50),
        },
        density_ranges={
            "small": (75, 145),
            "medium": (60, 115),
            "large": (40, 85),
            "mixed": (55, 135),
        },
    ),
}

MEDIA_PROFILES: dict[str, MediaProfile] = {
    "generic_dark_agar": MediaProfile(
        id="generic_dark_agar",
        display_name="Generic dark agar",
        aliases=("dark_agar", "generic agar"),
        agar_hex=DEFAULT_AGAR_HEX,
    ),
    "YPD": MediaProfile(
        id="YPD",
        display_name="YPD",
        aliases=("ypd", "yeast_peptone_dextrose", "yeast extract peptone dextrose"),
        agar_hex=DEFAULT_AGAR_HEX,
        use_cases=("yeast_growth",),
    ),
    "LSLB": MediaProfile(
        id="LSLB",
        display_name="LSLB / E. coli plate",
        aliases=("lslb", "lb", "luria_bertani", "lysogeny_broth"),
        agar_hex=ALT_AGAR_HEX,
        use_cases=("bacterial_plate", "selection_plate"),
    ),
}


@dataclass(frozen=True)
class Colony:
    x: int
    y: int
    radius: int
    cluster_id: int | None = None
    cluster_shape: str | None = None

    def yolo_label(self, image_size: int) -> str:
        diameter = self.radius * 2
        return (
            "0 "
            f"{self.x / image_size:.6f} "
            f"{self.y / image_size:.6f} "
            f"{diameter / image_size:.6f} "
            f"{diameter / image_size:.6f}"
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"x": self.x, "y": self.y, "radius": self.radius}
        if self.cluster_id is not None:
            payload["cluster_id"] = self.cluster_id
        if self.cluster_shape is not None:
            payload["cluster_shape"] = self.cluster_shape
        return payload


@dataclass
class SyntheticYoloPlate:
    image: np.ndarray
    labels: list[str]
    colonies: list[Colony]
    metadata: dict[str, Any]


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    normalized = hex_color.strip().lstrip("#")
    if len(normalized) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {hex_color!r}")
    try:
        red = int(normalized[0:2], 16)
        green = int(normalized[2:4], 16)
        blue = int(normalized[4:6], 16)
    except ValueError as exc:
        raise ValueError(f"Expected a valid hex color, got {hex_color!r}") from exc
    return (blue, green, red)


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def resolve_species_profile(species: str) -> SpeciesProfile:
    normalized = _normalize_key(species)
    for profile in SPECIES_PROFILES.values():
        keys = {_normalize_key(profile.id), *(_normalize_key(alias) for alias in profile.aliases)}
        if normalized in keys:
            return profile

    generic = SPECIES_PROFILES["generic_yeast"]
    return SpeciesProfile(
        id=species,
        display_name=species,
        aliases=(),
        colony_radius_ranges=_clone_ranges(generic.colony_radius_ranges),
        density_ranges=_clone_ranges(generic.density_ranges),
        colony_center_hex=generic.colony_center_hex,
        colony_edge_hex=generic.colony_edge_hex,
    )


def resolve_media_profile(medium: str) -> MediaProfile:
    normalized = _normalize_key(medium)
    for profile in MEDIA_PROFILES.values():
        keys = {_normalize_key(profile.id), *(_normalize_key(alias) for alias in profile.aliases)}
        if normalized in keys:
            return profile

    generic = MEDIA_PROFILES["generic_dark_agar"]
    return MediaProfile(
        id=medium,
        display_name=medium,
        aliases=(),
        agar_hex=generic.agar_hex,
        use_cases=generic.use_cases,
        base_profile=generic.id,
    )


def _validate_range_map(ranges: dict[str, tuple[int, int]], name: str) -> None:
    for size_mode, value in ranges.items():
        low, high = value
        if low <= 0 or high <= 0 or low > high:
            raise ValueError(f"{name}[{size_mode!r}] must be a positive (min, max) range")


def _validate_config(config: SyntheticDomainConfig) -> SyntheticDomainConfig:
    if config.image_size < 128:
        raise ValueError("image_size must be at least 128 pixels")
    dish_radius = config.dish_radius
    if dish_radius is None:
        dish_radius = int(config.image_size * DEFAULT_DISH_RADIUS_RATIO)
    if dish_radius <= 0 or dish_radius >= config.image_size // 2:
        raise ValueError("dish_radius must fit inside image_size")
    if config.collision_margin < 0:
        raise ValueError("collision_margin must be non-negative")
    offset_x, offset_y = config.dish_center_offset
    half_size = config.image_size // 2
    if abs(offset_x) + dish_radius >= half_size or abs(offset_y) + dish_radius >= half_size:
        raise ValueError("dish_center_offset must keep the dish fully inside image_size")

    hex_to_bgr(config.agar_hex)
    hex_to_bgr(config.colony_center_hex)
    hex_to_bgr(config.colony_edge_hex)
    _validate_range_map(config.colony_radius_ranges, "colony_radius_ranges")
    _validate_range_map(config.density_ranges, "density_ranges")
    return replace(config, dish_radius=int(dish_radius))


def build_domain_config(
    *,
    species: str = "generic_yeast",
    medium: str = "generic_dark_agar",
    image_size: int = DEFAULT_IMAGE_SIZE,
    dish_radius: int | None = None,
    random_seed: int | None = None,
    collision_margin: int = DEFAULT_COLLISION_MARGIN,
    dish_center_offset: tuple[int, int] = (0, 0),
    agar_hex: str | None = None,
    colony_center_hex: str | None = None,
    colony_edge_hex: str | None = None,
    colony_radius_ranges: dict[str, tuple[int, int]] | None = None,
    density_ranges: dict[str, tuple[int, int]] | None = None,
    selection_markers: tuple[str, ...] | list[str] = (),
) -> SyntheticDomainConfig:
    species_profile = resolve_species_profile(species)
    medium_profile = resolve_media_profile(medium)
    resolved_radius_ranges = _clone_ranges(species_profile.colony_radius_ranges)
    if colony_radius_ranges:
        resolved_radius_ranges.update(_clone_ranges(colony_radius_ranges))
    resolved_density_ranges = _clone_ranges(species_profile.density_ranges)
    if density_ranges:
        resolved_density_ranges.update(_clone_ranges(density_ranges))

    config = SyntheticDomainConfig(
        species=species_profile.id,
        medium=medium_profile.id,
        colony_radius_ranges=resolved_radius_ranges,
        colony_center_hex=colony_center_hex or species_profile.colony_center_hex,
        colony_edge_hex=colony_edge_hex or species_profile.colony_edge_hex,
        agar_hex=agar_hex or medium_profile.agar_hex,
        density_ranges=resolved_density_ranges,
        image_size=int(image_size),
        dish_radius=dish_radius,
        random_seed=random_seed,
        collision_margin=int(collision_margin),
        selection_markers=tuple(selection_markers),
        dish_center_offset=(int(dish_center_offset[0]), int(dish_center_offset[1])),
    )
    return _validate_config(config)


def _interpolate_bgr(
    start_bgr: tuple[int, int, int],
    end_bgr: tuple[int, int, int],
    amount: np.ndarray,
) -> np.ndarray:
    start = np.asarray(start_bgr, dtype=np.float32)
    end = np.asarray(end_bgr, dtype=np.float32)
    return start[None, None, :] * (1.0 - amount[:, :, None]) + end[None, None, :] * amount[:, :, None]


def _apply_agar_texture(
    image: np.ndarray,
    dish_mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    height, width = dish_mask.shape
    fine_noise = rng.normal(0.0, 3.0, size=(height, width)).astype(np.float32)
    broad_noise = rng.normal(0.0, 5.0, size=(height, width)).astype(np.float32)
    broad_noise = cv2.GaussianBlur(broad_noise, (0, 0), sigmaX=36.0, sigmaY=36.0)
    texture = fine_noise + broad_noise * 0.35

    dish_pixels = dish_mask > 0
    textured = image.astype(np.float32)
    textured[dish_pixels] += texture[dish_pixels][:, None]
    image[dish_pixels] = np.clip(textured[dish_pixels], 0, 255).astype(np.uint8)


def _draw_gradient_colony(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    center_bgr: tuple[int, int, int],
    edge_bgr: tuple[int, int, int],
    alpha: float,
) -> None:
    height, width = image.shape[:2]
    pad = radius + 3
    x0 = max(0, center[0] - pad)
    y0 = max(0, center[1] - pad)
    x1 = min(width, center[0] + pad + 1)
    y1 = min(height, center[1] + pad + 1)

    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2).astype(np.float32)
    draw_mask = distance <= float(radius) + 1.5
    if not np.any(draw_mask):
        return

    edge_amount = np.clip(distance / max(1.0, float(radius)), 0.0, 1.0)
    colony_color = _interpolate_bgr(center_bgr, edge_bgr, edge_amount)
    antialias = np.clip((float(radius) + 1.5 - distance) / 3.0, 0.0, 1.0)
    alpha_mask = (antialias * alpha)[:, :, None]

    blended = roi.astype(np.float32) * (1.0 - alpha_mask) + colony_color * alpha_mask
    roi[draw_mask] = np.clip(blended, 0, 255).astype(np.uint8)[draw_mask]


def _apply_lighting_stress(
    image: np.ndarray,
    rng: np.random.Generator,
    stress: ImageStressConfig,
) -> None:
    if (
        stress.brightness_delta == 0.0
        and stress.contrast == 1.0
        and stress.gradient_strength == 0.0
        and stress.vignette_strength == 0.0
    ):
        return

    height, width = image.shape[:2]
    adjusted = image.astype(np.float32)
    adjusted = (adjusted - 127.5) * float(stress.contrast) + 127.5 + float(stress.brightness_delta)

    if stress.gradient_strength:
        x_grid = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        y_grid = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        x_weight = float(rng.uniform(-1.0, 1.0))
        y_weight = float(rng.uniform(-1.0, 1.0))
        gradient = y_grid[:, None] * y_weight + x_grid[None, :] * x_weight
        adjusted += gradient[:, :, None] * float(stress.gradient_strength)

    if stress.vignette_strength:
        y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
        x_norm = (x_grid - width / 2.0) / max(1.0, width / 2.0)
        y_norm = (y_grid - height / 2.0) / max(1.0, height / 2.0)
        distance = np.sqrt(x_norm * x_norm + y_norm * y_norm)
        adjusted -= np.clip(distance, 0.0, 1.0)[:, :, None] * float(stress.vignette_strength)

    image[:, :] = np.clip(adjusted, 0, 255).astype(np.uint8)


def _apply_artifact_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    stress: ImageStressConfig,
) -> None:
    if not stress.artifact_noise:
        return

    height, width = image.shape[:2]
    min_count, max_count = stress.artifact_count_range
    artifact_count = int(rng.integers(min_count, max_count + 1)) if max_count > 0 else 0
    overlay = image.copy()

    for _ in range(artifact_count):
        kind = rng.choice(["speck", "scratch", "smudge"])
        color = tuple(int(value) for value in rng.integers(80, 235, size=3))
        if kind == "speck":
            center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            radius = int(rng.integers(2, max(3, width // 80)))
            cv2.circle(overlay, center, radius, color, -1, lineType=cv2.LINE_AA)
        elif kind == "scratch":
            start = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            end = (
                int(np.clip(start[0] + rng.integers(-width // 5, width // 5 + 1), 0, width - 1)),
                int(np.clip(start[1] + rng.integers(-height // 5, height // 5 + 1), 0, height - 1)),
            )
            cv2.line(overlay, start, end, color, int(rng.integers(1, 4)), lineType=cv2.LINE_AA)
        else:
            center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            axes = (int(rng.integers(8, max(9, width // 18))), int(rng.integers(4, max(5, height // 28))))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1, lineType=cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.22, image, 0.78, 0, dst=image)


def apply_image_stress(
    image: np.ndarray,
    rng: np.random.Generator,
    stress: ImageStressConfig,
) -> np.ndarray:
    stressed = image.copy()
    _apply_lighting_stress(stressed, rng, stress)
    _apply_artifact_noise(stressed, rng, stress)
    if stress.blur_sigma > 0:
        stressed = cv2.GaussianBlur(stressed, (0, 0), sigmaX=stress.blur_sigma, sigmaY=stress.blur_sigma)
    return stressed


def _collides(
    x_pos: int,
    y_pos: int,
    radius: int,
    existing_colonies: list[Colony],
    margin: int,
) -> bool:
    for colony in existing_colonies:
        distance = float(np.hypot(x_pos - colony.x, y_pos - colony.y))
        if distance < radius + colony.radius + margin:
            return True
    return False


def normalize_synthetic_schema(schema: str) -> str:
    normalized = _normalize_key(schema)
    if normalized in SYNTHETIC_SCHEMAS:
        return normalized
    raise ValueError(f"schema must be one of {', '.join(SYNTHETIC_SCHEMAS)}")


def _sample_merged_cluster_sizes(colony_count: int, rng: np.random.Generator) -> list[int]:
    min_size, max_size = MERGED_CLUSTER_SIZE_RANGE
    if colony_count < min_size:
        raise ValueError(f"{SCHEMA_MERGED_SNOWMAN} requires at least {min_size} colonies")

    cluster_count = int(np.ceil(colony_count / max_size))
    sizes = [min_size] * cluster_count
    remaining = colony_count - min_size * cluster_count
    order = list(rng.permutation(cluster_count))

    while remaining > 0:
        for index in order:
            if remaining <= 0:
                break
            capacity = max_size - sizes[index]
            if capacity <= 0:
                continue
            added = min(capacity, remaining)
            sizes[index] += added
            remaining -= added

    rng.shuffle(sizes)
    return sizes


def _merged_cluster_shape(cluster_size: int, cluster_id: int) -> str:
    if cluster_size == 2:
        return "snowman"
    if cluster_size == 3:
        return "tri_lobed" if cluster_id % 2 == 0 else "snowman"
    return "small_clump" if cluster_id % 2 == 0 else "tri_lobed"


def _touching_distance(radius_a: int, radius_b: int, rng: np.random.Generator) -> float:
    return float(radius_a + radius_b) * float(rng.uniform(0.68, 0.92))


def _center_offsets(offsets: list[tuple[float, float]]) -> list[tuple[float, float]]:
    center_x = sum(offset[0] for offset in offsets) / len(offsets)
    center_y = sum(offset[1] for offset in offsets) / len(offsets)
    return [(x_pos - center_x, y_pos - center_y) for x_pos, y_pos in offsets]


def _rotate_offsets(
    offsets: list[tuple[float, float]],
    angle: float,
) -> list[tuple[float, float]]:
    cos_angle = float(np.cos(angle))
    sin_angle = float(np.sin(angle))
    return [
        (
            x_pos * cos_angle - y_pos * sin_angle,
            x_pos * sin_angle + y_pos * cos_angle,
        )
        for x_pos, y_pos in offsets
    ]


def _sample_merged_cluster_offsets(
    *,
    radii: list[int],
    shape: str,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = [(0.0, 0.0)]
    if shape == "snowman":
        for index in range(1, len(radii)):
            previous_x, previous_y = offsets[index - 1]
            distance = _touching_distance(radii[index - 1], radii[index], rng)
            offsets.append(
                (
                    previous_x + distance,
                    previous_y + float(rng.uniform(-0.10, 0.10)) * min(radii[index - 1], radii[index]),
                )
            )
    elif shape == "tri_lobed":
        for index in range(1, len(radii)):
            angle = (2.0 * np.pi * (index - 1) / max(1, len(radii) - 1)) + float(rng.uniform(-0.22, 0.22))
            distance = _touching_distance(radii[0], radii[index], rng)
            offsets.append((distance * float(np.cos(angle)), distance * float(np.sin(angle))))
    else:
        base_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        for index in range(1, len(radii)):
            anchor_index = int(rng.integers(0, index))
            anchor_x, anchor_y = offsets[anchor_index]
            angle = base_angle + (2.0 * np.pi * index / len(radii)) + float(rng.uniform(-0.45, 0.45))
            distance = _touching_distance(radii[anchor_index], radii[index], rng)
            offsets.append((anchor_x + distance * float(np.cos(angle)), anchor_y + distance * float(np.sin(angle))))

    return _center_offsets(_rotate_offsets(offsets, float(rng.uniform(0.0, 2.0 * np.pi))))


def _merged_cluster_extent(offsets: list[tuple[float, float]], radii: list[int]) -> int:
    extent = max(float(np.hypot(x_pos, y_pos)) + float(radius) for (x_pos, y_pos), radius in zip(offsets, radii))
    return int(np.ceil(extent))


def _cluster_collides(
    x_pos: int,
    y_pos: int,
    radius: int,
    existing_clusters: list[tuple[int, int, int]],
    margin: int,
) -> bool:
    for cluster_x, cluster_y, cluster_radius in existing_clusters:
        distance = float(np.hypot(x_pos - cluster_x, y_pos - cluster_y))
        if distance < radius + cluster_radius + margin:
            return True
    return False


def _draw_soft_bridge(
    image: np.ndarray,
    first: Colony,
    second: Colony,
    edge_bgr: tuple[int, int, int],
) -> None:
    overlay = image.copy()
    thickness = max(2, int(round(min(first.radius, second.radius) * 1.35)))
    cv2.line(
        overlay,
        (first.x, first.y),
        (second.x, second.y),
        edge_bgr,
        thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.30, image, 0.70, 0, dst=image)


def _sample_colony_center(
    rng: np.random.Generator,
    dish_center: tuple[int, int],
    dish_radius: int,
    colony_radius: int,
    margin: int,
) -> tuple[int, int]:
    available_radius = max(1, dish_radius - colony_radius - margin)
    distance = float(np.sqrt(rng.uniform(0.0, 1.0)) * available_radius)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    return (
        int(round(dish_center[0] + distance * np.cos(angle))),
        int(round(dish_center[1] + distance * np.sin(angle))),
    )


def _generate_merged_colonies(
    *,
    image: np.ndarray,
    colony_count: int,
    min_radius: int,
    max_radius: int,
    rng: np.random.Generator,
    dish_center: tuple[int, int],
    dish_radius: int,
    collision_margin: int,
    max_attempts_multiplier: int,
    center_bgr: tuple[int, int, int],
    edge_bgr: tuple[int, int, int],
) -> tuple[list[Colony], list[dict[str, Any]]]:
    cluster_sizes = _sample_merged_cluster_sizes(colony_count, rng)
    colonies: list[Colony] = []
    clusters: list[dict[str, Any]] = []
    cluster_bounds: list[tuple[int, int, int]] = []
    attempts_per_cluster = max(80, max_attempts_multiplier)

    for cluster_id, cluster_size in enumerate(cluster_sizes):
        placed_cluster = False
        shape = _merged_cluster_shape(cluster_size, cluster_id)

        for _ in range(attempts_per_cluster):
            radii = [int(rng.integers(min_radius, max_radius + 1)) for _ in range(cluster_size)]
            offsets = _sample_merged_cluster_offsets(radii=radii, shape=shape, rng=rng)
            cluster_radius = _merged_cluster_extent(offsets, radii)
            cluster_x, cluster_y = _sample_colony_center(
                rng=rng,
                dish_center=dish_center,
                dish_radius=dish_radius,
                colony_radius=cluster_radius,
                margin=collision_margin + 2,
            )
            if _cluster_collides(
                cluster_x,
                cluster_y,
                cluster_radius,
                cluster_bounds,
                margin=max(1, collision_margin),
            ):
                continue

            candidate_colonies = [
                Colony(
                    x=int(round(cluster_x + offset_x)),
                    y=int(round(cluster_y + offset_y)),
                    radius=radius,
                    cluster_id=cluster_id,
                    cluster_shape=shape,
                )
                for (offset_x, offset_y), radius in zip(offsets, radii)
            ]
            if any(
                float(np.hypot(colony.x - dish_center[0], colony.y - dish_center[1])) + colony.radius
                > dish_radius - 1
                for colony in candidate_colonies
            ):
                continue

            for first_index, first in enumerate(candidate_colonies):
                for second in candidate_colonies[first_index + 1 :]:
                    distance = float(np.hypot(first.x - second.x, first.y - second.y))
                    if distance <= (first.radius + second.radius) * 1.05:
                        _draw_soft_bridge(image, first, second, edge_bgr)

            for colony in candidate_colonies:
                _draw_gradient_colony(
                    image=image,
                    center=(colony.x, colony.y),
                    radius=colony.radius,
                    center_bgr=center_bgr,
                    edge_bgr=edge_bgr,
                    alpha=float(rng.uniform(0.90, 0.98)),
                )

            start_index = len(colonies)
            colonies.extend(candidate_colonies)
            clusters.append(
                {
                    "id": cluster_id,
                    "shape": shape,
                    "x": cluster_x,
                    "y": cluster_y,
                    "radius": cluster_radius,
                    "colony_count": len(candidate_colonies),
                    "colony_indices": list(range(start_index, len(colonies))),
                }
            )
            cluster_bounds.append((cluster_x, cluster_y, cluster_radius))
            placed_cluster = True
            break

        if not placed_cluster:
            LOGGER.warning(
                "Only placed %s of %s requested %s colonies before merged cluster placement saturated.",
                len(colonies),
                colony_count,
                SCHEMA_MERGED_SNOWMAN,
            )
            break

    return colonies, clusters


def _profile_payload(config: SyntheticDomainConfig) -> dict[str, Any]:
    return {
        "synthetic_config": asdict(config),
        "species_profile": asdict(resolve_species_profile(config.species)),
        "medium_profile": asdict(resolve_media_profile(config.medium)),
    }


def generate_plate(
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    colony_count: int = 80,
    size_mode: str = "mixed",
    schema: str = DEFAULT_SYNTHETIC_SCHEMA,
    rng: np.random.Generator | None = None,
    agar_hex: str | None = None,
    collision_margin: int | None = None,
    max_attempts_multiplier: int = 80,
    config: SyntheticDomainConfig | None = None,
    dish_radius: int | None = None,
) -> SyntheticYoloPlate:
    rng = rng or np.random.default_rng()
    schema = normalize_synthetic_schema(schema)
    if config is None:
        config = build_domain_config(
            image_size=image_size,
            dish_radius=dish_radius,
            agar_hex=agar_hex,
            collision_margin=DEFAULT_COLLISION_MARGIN if collision_margin is None else collision_margin,
        )
    else:
        config = _validate_config(config)
        if agar_hex is not None:
            config = replace(config, agar_hex=agar_hex)
        if dish_radius is not None:
            config = replace(config, dish_radius=dish_radius)
        if collision_margin is not None:
            config = replace(config, collision_margin=collision_margin)
        config = _validate_config(config)

    if size_mode not in config.colony_radius_ranges:
        raise ValueError(f"Unknown size mode {size_mode!r}; expected one of {sorted(config.colony_radius_ranges)}")
    if colony_count < 0:
        raise ValueError("colony_count must be non-negative")

    image_size = config.image_size
    dish_center = (
        image_size // 2 + int(config.dish_center_offset[0]),
        image_size // 2 + int(config.dish_center_offset[1]),
    )
    dish_radius = int(config.dish_radius or image_size * DEFAULT_DISH_RADIUS_RATIO)
    agar_bgr = hex_to_bgr(config.agar_hex)
    outline_bgr = hex_to_bgr(DISH_OUTLINE_HEX)
    center_bgr = hex_to_bgr(config.colony_center_hex)
    edge_bgr = hex_to_bgr(config.colony_edge_hex)

    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    dish_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.circle(dish_mask, dish_center, dish_radius, 255, -1, lineType=cv2.LINE_AA)
    image[dish_mask > 0] = agar_bgr
    _apply_agar_texture(image, dish_mask, rng)

    min_radius, max_radius = config.colony_radius_ranges[size_mode]
    colonies: list[Colony] = []
    merged_clusters: list[dict[str, Any]] = []
    max_attempts = max(colony_count * max_attempts_multiplier, 200)

    if schema == SCHEMA_MERGED_SNOWMAN:
        colonies, merged_clusters = _generate_merged_colonies(
            image=image,
            colony_count=colony_count,
            min_radius=min_radius,
            max_radius=max_radius,
            rng=rng,
            dish_center=dish_center,
            dish_radius=dish_radius,
            collision_margin=config.collision_margin,
            max_attempts_multiplier=max_attempts_multiplier,
            center_bgr=center_bgr,
            edge_bgr=edge_bgr,
        )
    else:
        for _ in range(max_attempts):
            if len(colonies) >= colony_count:
                break

            radius = int(rng.integers(min_radius, max_radius + 1))
            x_pos, y_pos = _sample_colony_center(
                rng=rng,
                dish_center=dish_center,
                dish_radius=dish_radius,
                colony_radius=radius,
                margin=config.collision_margin + 2,
            )
            if _collides(x_pos, y_pos, radius, colonies, margin=config.collision_margin):
                continue

            colonies.append(Colony(x=x_pos, y=y_pos, radius=radius))
            _draw_gradient_colony(
                image=image,
                center=(x_pos, y_pos),
                radius=radius,
                center_bgr=center_bgr,
                edge_bgr=edge_bgr,
                alpha=float(rng.uniform(0.88, 0.97)),
            )

    if len(colonies) < colony_count:
        LOGGER.warning(
            "Only placed %s of %s requested %s colonies at %sx%s.",
            len(colonies),
            colony_count,
            size_mode,
            image_size,
            image_size,
        )

    cv2.circle(image, dish_center, dish_radius, outline_bgr, 3, lineType=cv2.LINE_AA)
    labels = [colony.yolo_label(image_size) for colony in colonies]
    colors = {
        "background_hex": BLACK_BACKGROUND_HEX,
        "agar_hex": config.agar_hex,
        "dish_outline_hex": DISH_OUTLINE_HEX,
        "colony_center_hex": config.colony_center_hex,
        "colony_edge_hex": config.colony_edge_hex,
    }
    radius_range = list(config.colony_radius_ranges[size_mode])
    density_range = list(config.density_ranges.get(size_mode, (colony_count, colony_count)))
    metadata: dict[str, Any] = {
        "schema": schema,
        "image_size": image_size,
        "size_mode": size_mode,
        "requested_colonies": colony_count,
        "placed_colonies": len(colonies),
        "colony_count": len(colonies),
        "species": config.species,
        "medium": config.medium,
        "selection_markers": list(config.selection_markers),
        "dish": {"x": dish_center[0], "y": dish_center[1], "radius": dish_radius},
        "colors": colors,
        "radius_range": radius_range,
        "density_range": density_range,
        "collision_margin": config.collision_margin,
        "colonies": [colony.to_dict() for colony in colonies],
        "generator_parameters": {
            "generator_version": GENERATOR_VERSION,
            "schema": schema,
            "species": config.species,
            "medium": config.medium,
            "image_size": image_size,
            "dish_radius": dish_radius,
            "dish_center_offset": list(config.dish_center_offset),
            "size_mode": size_mode,
            "requested_colonies": colony_count,
            "colony_count": len(colonies),
            "colony_radius_range": radius_range,
            "density_range": density_range,
            "collision_margin": config.collision_margin,
            "selection_markers": list(config.selection_markers),
            "colors": colors,
            "max_attempts_multiplier": max_attempts_multiplier,
        },
    }
    if schema == SCHEMA_MERGED_SNOWMAN:
        metadata.update(
            {
                "placed_clusters": len(merged_clusters),
                "cluster_size_range": list(MERGED_CLUSTER_SIZE_RANGE),
                "merged_clusters": merged_clusters,
            }
        )
        metadata["generator_parameters"].update(
            {
                "cluster_size_range": list(MERGED_CLUSTER_SIZE_RANGE),
                "placed_clusters": len(merged_clusters),
                "merged_cluster_shapes": sorted({str(cluster["shape"]) for cluster in merged_clusters}),
            }
        )
    return SyntheticYoloPlate(image=image, labels=labels, colonies=colonies, metadata=metadata)


def save_yolo_plate(
    plate: SyntheticYoloPlate,
    image_path: Path,
    label_path: Path,
    *,
    jpeg_quality: int = 94,
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), plate.image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    label_path.write_text("\n".join(plate.labels) + ("\n" if plate.labels else ""), encoding="utf-8")


def build_size_mode_sequence(plates: int, size_mode: str = "starter") -> list[str]:
    if plates <= 0:
        raise ValueError("plates must be greater than zero")
    if size_mode in SIZE_RANGES:
        return [size_mode] * plates
    if size_mode != "starter":
        raise ValueError("size_mode must be one of small, medium, large, mixed, or starter")

    modes = ["small", "medium", "large", "mixed"]
    base_count, remainder = divmod(plates, len(modes))
    sequence: list[str] = []
    for index, mode in enumerate(modes):
        sequence.extend([mode] * (base_count + (1 if index < remainder else 0)))
    return sequence


def write_dataset_yaml(dataset_root: Path) -> Path:
    dataset_yaml = "\n".join(
        [
            f"path: {dataset_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "nc: 1",
            f'names: ["{CLASS_NAME}"]',
            "",
        ]
    )
    yaml_path = dataset_root / "dataset.yaml"
    yaml_path.write_text(dataset_yaml, encoding="utf-8")
    return yaml_path


def write_flat_split_dataset_yaml(split_dir: Path) -> Path:
    dataset_yaml = "\n".join(
        [
            f"path: {split_dir.resolve()}",
            "train: images",
            "val: images",
            "nc: 1",
            f'names: ["{CLASS_NAME}"]',
            "",
        ]
    )
    yaml_path = split_dir / "dataset.yaml"
    yaml_path.write_text(dataset_yaml, encoding="utf-8")
    return yaml_path


def write_suite_training_yaml(suite_dir: Path) -> Path:
    dataset_yaml = "\n".join(
        [
            f"path: {suite_dir.resolve()}",
            "train: train_standard/images",
            "val: val_standard/images",
            "nc: 1",
            f'names: ["{CLASS_NAME}"]',
            "",
        ]
    )
    yaml_path = suite_dir / "dataset.yaml"
    yaml_path.write_text(dataset_yaml, encoding="utf-8")
    return yaml_path


def _scale_ranges_for_image_size(
    ranges: dict[str, tuple[int, int]],
    image_size: int,
    *,
    floor: int = 1,
) -> dict[str, tuple[int, int]]:
    if image_size == DEFAULT_IMAGE_SIZE:
        return _clone_ranges(ranges)
    scale = (image_size / DEFAULT_IMAGE_SIZE) ** 2
    scaled: dict[str, tuple[int, int]] = {}
    for key, (low, high) in ranges.items():
        scaled_low = max(floor, int(round(low * scale)))
        scaled_high = max(scaled_low, int(round(high * scale)))
        scaled[key] = (scaled_low, scaled_high)
    return scaled


def _balanced_modes(plates: int, modes: tuple[str, ...]) -> list[str]:
    if plates <= 0:
        return []
    base_count, remainder = divmod(plates, len(modes))
    sequence: list[str] = []
    for index, mode in enumerate(modes):
        sequence.extend([mode] * (base_count + (1 if index < remainder else 0)))
    return sequence


def _split_mode_sequence(plates: int, size_mode: str) -> list[str]:
    if size_mode == "size_extremes":
        return _balanced_modes(plates, ("small", "large"))
    return build_size_mode_sequence(plates, size_mode=size_mode)


def _sample_dish_offset(
    rng: np.random.Generator,
    *,
    image_size: int,
    dish_radius: int,
    offset_fraction_range: tuple[float, float],
) -> tuple[int, int]:
    low_fraction, high_fraction = offset_fraction_range
    if high_fraction <= 0:
        return (0, 0)

    max_safe_offset = max(0, image_size // 2 - dish_radius - 2)
    low = max(0.0, float(low_fraction)) * image_size
    high = max(low, float(high_fraction) * image_size)
    distance = min(float(rng.uniform(low, high)), float(max_safe_offset))
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    return (int(round(distance * np.cos(angle))), int(round(distance * np.sin(angle))))


def default_suite_specs(
    *,
    train_plates: int = 80,
    val_plates: int = 20,
    stress_plates: int = 20,
    image_size: int = DEFAULT_IMAGE_SIZE,
    species: str = "generic_yeast",
    medium: str = "generic_dark_agar",
) -> list[SyntheticSplitSpec]:
    standard_density = _scale_ranges_for_image_size(DEFAULT_COLONY_RANGES, image_size)
    density_extremes = _scale_ranges_for_image_size(
        {
            "small": (130, 190),
            "medium": (5, 22),
            "large": (3, 14),
            "mixed": (8, 170),
        },
        image_size,
    )
    size_extreme_density = _scale_ranges_for_image_size(
        {
            "small": (80, 140),
            "medium": (40, 80),
            "large": (12, 28),
            "mixed": (24, 90),
        },
        image_size,
    )

    return [
        SyntheticSplitSpec(
            name="train_standard",
            tier="train",
            plates=train_plates,
            size_mode="starter",
            species=species,
            medium=medium,
            density_ranges=standard_density,
            notes="Standard synthetic training distribution.",
        ),
        SyntheticSplitSpec(
            name="val_standard",
            tier="same_generator_validation",
            plates=val_plates,
            size_mode="starter",
            species=species,
            medium=medium,
            density_ranges=standard_density,
            notes="Same-generator validation distribution; not real-world validation.",
        ),
        SyntheticSplitSpec(
            name="test_lighting_shift",
            tier="ood_synthetic_stress",
            plates=stress_plates,
            size_mode="starter",
            species=species,
            medium=medium,
            density_ranges=standard_density,
            image_stress=ImageStressConfig(
                brightness_delta=-18.0,
                contrast=1.12,
                gradient_strength=28.0,
                vignette_strength=24.0,
            ),
            notes="Brightness, contrast, gradient, and vignette changes.",
        ),
        SyntheticSplitSpec(
            name="test_agar_color_shift",
            tier="ood_synthetic_stress",
            plates=stress_plates,
            size_mode="starter",
            species=species,
            medium=medium,
            agar_hex=ALT_AGAR_HEX,
            density_ranges=standard_density,
            notes="Agar base color shifted from the standard dark agar.",
        ),
        SyntheticSplitSpec(
            name="test_blur_compression",
            tier="ood_synthetic_stress",
            plates=stress_plates,
            size_mode="starter",
            species=species,
            medium=medium,
            density_ranges=standard_density,
            image_stress=ImageStressConfig(blur_sigma=1.4, jpeg_quality=48),
            notes="Defocus blur plus lower-quality JPEG compression.",
        ),
        SyntheticSplitSpec(
            name="test_density_extremes",
            tier="ood_synthetic_stress",
            plates=stress_plates,
            size_mode="starter",
            species=species,
            medium=medium,
            density_ranges=density_extremes,
            notes="Sparse and dense colony-count regimes outside the standard distribution.",
        ),
        SyntheticSplitSpec(
            name="test_size_extremes",
            tier="ood_synthetic_stress",
            plates=stress_plates,
            size_mode="size_extremes",
            species=species,
            medium=medium,
            colony_radius_ranges={
                "small": (8, 14),
                "medium": (18, 30),
                "large": (62, 85),
                "mixed": (8, 85),
            },
            density_ranges=size_extreme_density,
            notes="Very small and very large colonies relative to the standard synthetic set.",
        ),
        SyntheticSplitSpec(
            name="test_plate_position_shift",
            tier="ood_synthetic_stress",
            plates=stress_plates,
            size_mode="starter",
            species=species,
            medium=medium,
            density_ranges=standard_density,
            dish_offset_fraction_range=(0.04, 0.09),
            notes="Petri dish shifted away from image center.",
        ),
        SyntheticSplitSpec(
            name="test_artifact_noise",
            tier="ood_synthetic_stress",
            plates=stress_plates,
            size_mode="starter",
            species=species,
            medium=medium,
            density_ranges=standard_density,
            image_stress=ImageStressConfig(
                artifact_noise=True,
                artifact_count_range=(18, 42),
                brightness_delta=6.0,
                contrast=0.96,
            ),
            notes="Synthetic dust, scratches, and smudges over the image.",
        ),
    ]


def _train_count_for_split(plates: int, train_ratio: float) -> int:
    if plates == 1:
        return 1
    train_count = int(round(plates * train_ratio))
    return max(1, min(train_count, plates - 1))


def generate_dataset(
    *,
    output_dir: Path,
    plates: int = 100,
    image_size: int = DEFAULT_IMAGE_SIZE,
    train_ratio: float = 0.8,
    seed: int = 17,
    size_mode: str = "starter",
    schema: str = DEFAULT_SYNTHETIC_SCHEMA,
    species: str = "generic_yeast",
    medium: str = "generic_dark_agar",
    agar_hex: str | None = None,
    colony_center_hex: str | None = None,
    colony_edge_hex: str | None = None,
    dish_radius: int | None = None,
    collision_margin: int = DEFAULT_COLLISION_MARGIN,
    selection_markers: tuple[str, ...] | list[str] = (),
    overwrite: bool = False,
    colony_count_ranges: dict[str, tuple[int, int]] | None = None,
    colony_radius_ranges: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    schema = normalize_synthetic_schema(schema)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists and is not empty; pass overwrite=True to replace it")
        shutil.rmtree(output_dir)

    config = build_domain_config(
        species=species,
        medium=medium,
        image_size=image_size,
        dish_radius=dish_radius,
        random_seed=seed,
        collision_margin=collision_margin,
        agar_hex=agar_hex,
        colony_center_hex=colony_center_hex,
        colony_edge_hex=colony_edge_hex,
        colony_radius_ranges=colony_radius_ranges,
        density_ranges=colony_count_ranges,
        selection_markers=selection_markers,
    )

    rng = np.random.default_rng(seed)
    mode_sequence = build_size_mode_sequence(plates, size_mode=size_mode)
    rng.shuffle(mode_sequence)
    train_count = _train_count_for_split(plates, train_ratio)
    image_records: list[dict[str, Any]] = []

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for index, current_mode in enumerate(mode_sequence):
        split = "train" if index < train_count else "val"
        colony_min, colony_max = config.density_ranges[current_mode]
        requested_colonies = int(rng.integers(colony_min, colony_max + 1))
        plate = generate_plate(
            colony_count=requested_colonies,
            size_mode=current_mode,
            schema=schema,
            rng=rng,
            config=config,
        )

        stem_prefix = "apricot" if schema == SCHEMA_CLEAN_DOTS else f"apricot_{schema}"
        stem = f"{stem_prefix}_{current_mode}_{index:05d}"
        image_path = output_dir / "images" / split / f"{stem}.jpg"
        label_path = output_dir / "labels" / split / f"{stem}.txt"
        save_yolo_plate(plate, image_path, label_path)

        record = dict(plate.metadata)
        generator_parameters = dict(record["generator_parameters"])
        generator_parameters.update(
            {
                "random_seed": seed,
                "plate_index": index,
                "split": split,
                "train_ratio": train_ratio,
                "jpeg_quality": 94,
            }
        )
        record.update(
            {
                "image": str(image_path.relative_to(output_dir)),
                "label": str(label_path.relative_to(output_dir)),
                "split": split,
                "stem": stem,
                "generator_parameters": generator_parameters,
            }
        )
        image_records.append(record)

    yaml_path = write_dataset_yaml(output_dir)
    profile_payload = _profile_payload(config)
    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "schema": schema,
        "schema_labels": list(PLATE_SCHEMA_LABELS),
        "plate_schema_registry": plate_schema_registry_payload(),
        "random_seed": seed,
        "species_profile": profile_payload["species_profile"],
        "medium_profile": profile_payload["medium_profile"],
        "synthetic_config": profile_payload["synthetic_config"],
        "number_of_images": plates,
        "train_val_split": {
            "train_ratio": train_ratio,
            "train": train_count,
            "val": plates - train_count,
        },
        "colony_count_range": config.density_ranges,
        "colony_count_range_overall": [
            min(value[0] for value in config.density_ranges.values()),
            max(value[1] for value in config.density_ranges.values()),
        ],
        "radius_ranges": config.colony_radius_ranges,
        "no_overlap_margin": config.collision_margin,
        "class_names": CLASS_NAMES,
        "output_path": str(output_dir.resolve()),
        "images": image_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    counts_by_mode = {mode: mode_sequence.count(mode) for mode in SIZE_RANGES}
    return {
        "dataset": str(output_dir),
        "dataset_yaml": str(yaml_path),
        "manifest": str(manifest_path),
        "plates": plates,
        "train": train_count,
        "val": plates - train_count,
        "image_size": config.image_size,
        "dish_radius": config.dish_radius,
        "species": config.species,
        "medium": config.medium,
        "schema": schema,
        "class_name": CLASS_NAME,
        "counts_by_mode": counts_by_mode,
        "total_colonies": sum(int(record["placed_colonies"]) for record in image_records),
    }


def generate_synthetic_split(
    *,
    split_dir: Path,
    spec: SyntheticSplitSpec,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = 17,
    dish_radius: int | None = None,
    collision_margin: int = DEFAULT_COLLISION_MARGIN,
    selection_markers: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    if spec.plates <= 0:
        raise ValueError(f"{spec.name} must contain at least one image")

    rng = np.random.default_rng(seed)
    split_dir.mkdir(parents=True, exist_ok=True)
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    base_config = build_domain_config(
        species=spec.species,
        medium=spec.medium,
        image_size=image_size,
        dish_radius=dish_radius,
        random_seed=seed,
        collision_margin=collision_margin,
        agar_hex=spec.agar_hex,
        colony_center_hex=spec.colony_center_hex,
        colony_edge_hex=spec.colony_edge_hex,
        colony_radius_ranges=spec.colony_radius_ranges,
        density_ranges=spec.density_ranges,
        selection_markers=selection_markers,
    )

    mode_sequence = _split_mode_sequence(spec.plates, spec.size_mode)
    rng.shuffle(mode_sequence)
    image_records: list[dict[str, Any]] = []

    for index, current_mode in enumerate(mode_sequence):
        offset = _sample_dish_offset(
            rng,
            image_size=base_config.image_size,
            dish_radius=int(base_config.dish_radius or base_config.image_size * DEFAULT_DISH_RADIUS_RATIO),
            offset_fraction_range=spec.dish_offset_fraction_range,
        )
        plate_config = _validate_config(replace(base_config, dish_center_offset=offset))
        colony_min, colony_max = plate_config.density_ranges[current_mode]
        requested_colonies = int(rng.integers(colony_min, colony_max + 1))
        plate = generate_plate(
            colony_count=requested_colonies,
            size_mode=current_mode,
            rng=rng,
            config=plate_config,
        )
        if spec.image_stress != ImageStressConfig():
            plate.image = apply_image_stress(plate.image, rng, spec.image_stress)
            plate.metadata["image_stress"] = asdict(spec.image_stress)
        else:
            plate.metadata["image_stress"] = asdict(spec.image_stress)

        stem = f"{spec.name}_{current_mode}_{index:05d}"
        image_path = images_dir / f"{stem}.jpg"
        label_path = labels_dir / f"{stem}.txt"
        save_yolo_plate(
            plate,
            image_path,
            label_path,
            jpeg_quality=spec.image_stress.jpeg_quality,
        )

        expected_count = len(plate.labels)
        record = dict(plate.metadata)
        generator_parameters = dict(record["generator_parameters"])
        generator_parameters.update(
            {
                "random_seed": seed,
                "plate_index": index,
                "split": spec.name,
                "tier": spec.tier,
                "dish_center_offset": list(offset),
                "image_stress": asdict(spec.image_stress),
                "jpeg_quality": spec.image_stress.jpeg_quality,
            }
        )
        record.update(
            {
                "image": str(image_path.relative_to(split_dir)),
                "label": str(label_path.relative_to(split_dir)),
                "split": spec.name,
                "tier": spec.tier,
                "stem": stem,
                "expected_colony_count": expected_count,
                "label_count": expected_count,
                "generator_parameters": generator_parameters,
            }
        )
        image_records.append(record)

    dataset_yaml = write_flat_split_dataset_yaml(split_dir)
    profile_payload = _profile_payload(base_config)
    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "schema": DEFAULT_SYNTHETIC_SCHEMA,
        "schema_labels": list(PLATE_SCHEMA_LABELS),
        "plate_schema_registry": plate_schema_registry_payload(),
        "split": spec.name,
        "tier": spec.tier,
        "notes": spec.notes,
        "random_seed": seed,
        "species_profile": profile_payload["species_profile"],
        "medium_profile": profile_payload["medium_profile"],
        "synthetic_config": profile_payload["synthetic_config"],
        "split_spec": asdict(spec),
        "number_of_images": spec.plates,
        "class_names": CLASS_NAMES,
        "dataset_yaml": str(dataset_yaml.relative_to(split_dir)),
        "expected_counts": [
            {
                "image": record["image"],
                "label": record["label"],
                "schema": record["schema"],
                "species": record["species"],
                "medium": record["medium"],
                "colony_count": record["colony_count"],
                "expected_colony_count": record["expected_colony_count"],
            }
            for record in image_records
        ],
        "images": image_records,
    }
    manifest_path = split_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "name": spec.name,
        "tier": spec.tier,
        "path": str(split_dir),
        "manifest": str(manifest_path),
        "dataset_yaml": str(dataset_yaml),
        "schema": DEFAULT_SYNTHETIC_SCHEMA,
        "images": spec.plates,
        "total_colonies": sum(int(record["expected_colony_count"]) for record in image_records),
    }


def generate_synthetic_suite(
    *,
    output_dir: Path,
    train_plates: int = 80,
    val_plates: int = 20,
    stress_plates: int = 20,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = 17,
    species: str = "generic_yeast",
    medium: str = "generic_dark_agar",
    dish_radius: int | None = None,
    collision_margin: int = DEFAULT_COLLISION_MARGIN,
    selection_markers: tuple[str, ...] | list[str] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists and is not empty; pass overwrite=True to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_specs = default_suite_specs(
        train_plates=train_plates,
        val_plates=val_plates,
        stress_plates=stress_plates,
        image_size=image_size,
        species=species,
        medium=medium,
    )

    split_summaries: list[dict[str, Any]] = []
    for index, spec in enumerate(split_specs):
        split_seed = seed + index * 10_003
        split_summaries.append(
            generate_synthetic_split(
                split_dir=output_dir / spec.name,
                spec=spec,
                image_size=image_size,
                seed=split_seed,
                dish_radius=dish_radius,
                collision_margin=collision_margin,
                selection_markers=selection_markers,
            )
        )

    training_yaml = write_suite_training_yaml(output_dir)
    suite_manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "schema": DEFAULT_SYNTHETIC_SCHEMA,
        "schema_labels": list(PLATE_SCHEMA_LABELS),
        "plate_schema_registry": plate_schema_registry_payload(),
        "protocol": "synthetic_only_three_tier",
        "warning": (
            "Synthetic train/validation/stress results are not real-world validation. "
            "They only measure behavior against known synthetic regimes."
        ),
        "random_seed": seed,
        "image_size": image_size,
        "class_names": CLASS_NAMES,
        "training_dataset_yaml": str(training_yaml.relative_to(output_dir)),
        "tiers": {
            "train": "train_standard",
            "same_generator_validation": "val_standard",
            "ood_synthetic_stress": list(STRESS_SPLITS),
        },
        "splits": split_summaries,
    }
    suite_manifest_path = output_dir / "suite_manifest.json"
    suite_manifest_path.write_text(json.dumps(suite_manifest, indent=2), encoding="utf-8")

    return {
        "suite": str(output_dir),
        "suite_manifest": str(suite_manifest_path),
        "dataset_yaml": str(training_yaml),
        "splits": split_summaries,
        "schema_labels": list(PLATE_SCHEMA_LABELS),
        "total_images": sum(int(summary["images"]) for summary in split_summaries),
        "total_colonies": sum(int(summary["total_colonies"]) for summary in split_summaries),
    }
