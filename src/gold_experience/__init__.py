from __future__ import annotations

from typing import Any

__all__ = ["ColonyAnnotation", "ColonyCounterPipeline", "PlateGoldAnnotation", "PlateUserHints"]


def __getattr__(name: str) -> Any:
    if name == "ColonyCounterPipeline":
        from .pipeline import ColonyCounterPipeline

        return ColonyCounterPipeline
    if name in {"ColonyAnnotation", "PlateGoldAnnotation", "PlateUserHints"}:
        from .annotations import ColonyAnnotation, PlateGoldAnnotation, PlateUserHints

        return {
            "ColonyAnnotation": ColonyAnnotation,
            "PlateGoldAnnotation": PlateGoldAnnotation,
            "PlateUserHints": PlateUserHints,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
