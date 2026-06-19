from __future__ import annotations

from dataclasses import asdict, dataclass

SCHEMA_CLEAN_DOTS = "clean_dots"
SCHEMA_MERGED_SNOWMAN = "merged_snowman"
SCHEMA_STREAK_LINES = "streak_lines"
SCHEMA_MIXED_PLATE = "mixed_plate"
SCHEMA_UNKNOWN = "unknown"
PLATE_SCHEMA_LABELS = (
    SCHEMA_CLEAN_DOTS,
    SCHEMA_MERGED_SNOWMAN,
    SCHEMA_STREAK_LINES,
    SCHEMA_MIXED_PLATE,
    SCHEMA_UNKNOWN,
)
DEFAULT_SYNTHETIC_SCHEMA = SCHEMA_CLEAN_DOTS


@dataclass(frozen=True)
class PlateSchema:
    label: str
    display_name: str
    description: str


PLATE_SCHEMA_REGISTRY: dict[str, PlateSchema] = {
    SCHEMA_CLEAN_DOTS: PlateSchema(
        label=SCHEMA_CLEAN_DOTS,
        display_name="Clean dots",
        description="Separated round colonies with no intentional merges or streak-line morphology.",
    ),
    SCHEMA_MERGED_SNOWMAN: PlateSchema(
        label=SCHEMA_MERGED_SNOWMAN,
        display_name="Merged snowman",
        description="Touching or partially merged round colonies, including two-lobed snowman-like clusters.",
    ),
    SCHEMA_STREAK_LINES: PlateSchema(
        label=SCHEMA_STREAK_LINES,
        display_name="Streak lines",
        description="Linear streak growth or dragged colony morphology.",
    ),
    SCHEMA_MIXED_PLATE: PlateSchema(
        label=SCHEMA_MIXED_PLATE,
        display_name="Mixed plate",
        description="A plate containing more than one registered morphology schema.",
    ),
    SCHEMA_UNKNOWN: PlateSchema(
        label=SCHEMA_UNKNOWN,
        display_name="Unknown",
        description="Schema could not be determined or has not yet been mapped into the registry.",
    ),
}


def plate_schema_registry_payload() -> dict[str, dict[str, str]]:
    return {label: asdict(PLATE_SCHEMA_REGISTRY[label]) for label in PLATE_SCHEMA_LABELS}
