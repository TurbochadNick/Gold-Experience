from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DishAnnotation:
    cx: float
    cy: float
    rx: float
    ry: float
    rotation: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "cx": float(self.cx),
            "cy": float(self.cy),
            "rx": float(self.rx),
            "ry": float(self.ry),
            "rotation": float(self.rotation),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DishAnnotation":
        return cls(
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
            rx=float(payload["rx"]),
            ry=float(payload["ry"]),
            rotation=float(payload.get("rotation", 0.0)),
        )


@dataclass
class PointAnnotation:
    x: float
    y: float
    radius: float | None = None
    tag: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"x": float(self.x), "y": float(self.y)}
        if self.radius is not None:
            payload["radius"] = float(self.radius)
        if self.tag:
            payload["tag"] = str(self.tag)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PointAnnotation":
        return cls(
            x=float(payload["x"]),
            y=float(payload["y"]),
            radius=None if "radius" not in payload else float(payload["radius"]),
            tag=payload.get("tag"),
        )


@dataclass
class PolygonAnnotation:
    points: list[tuple[float, float]]
    tag: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "points": [[float(x_pos), float(y_pos)] for x_pos, y_pos in self.points]
        }
        if self.tag:
            payload["tag"] = str(self.tag)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolygonAnnotation":
        return cls(
            points=[(float(x_pos), float(y_pos)) for x_pos, y_pos in payload["points"]],
            tag=payload.get("tag"),
        )


@dataclass
class PlateGoldAnnotation:
    image: str
    image_width: int
    image_height: int
    dish: DishAnnotation | None = None
    colonies: list[PointAnnotation] = field(default_factory=list)
    label_regions: list[PolygonAnnotation] = field(default_factory=list)
    ignore_regions: list[PolygonAnnotation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "dish": None if self.dish is None else self.dish.to_dict(),
            "colonies": [item.to_dict() for item in self.colonies],
            "label_regions": [item.to_dict() for item in self.label_regions],
            "ignore_regions": [item.to_dict() for item in self.ignore_regions],
            "metadata": dict(self.metadata),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlateGoldAnnotation":
        return cls(
            image=str(payload["image"]),
            image_width=int(payload["image_width"]),
            image_height=int(payload["image_height"]),
            dish=None if payload.get("dish") is None else DishAnnotation.from_dict(payload["dish"]),
            colonies=[PointAnnotation.from_dict(item) for item in payload.get("colonies", [])],
            label_regions=[PolygonAnnotation.from_dict(item) for item in payload.get("label_regions", [])],
            ignore_regions=[PolygonAnnotation.from_dict(item) for item in payload.get("ignore_regions", [])],
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: Path) -> "PlateGoldAnnotation":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class PlateUserHints:
    image: str
    image_width: int
    image_height: int
    positive_clicks: list[PointAnnotation] = field(default_factory=list)
    negative_clicks: list[PointAnnotation] = field(default_factory=list)
    label_hints: list[PolygonAnnotation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "positive_clicks": [item.to_dict() for item in self.positive_clicks],
            "negative_clicks": [item.to_dict() for item in self.negative_clicks],
            "label_hints": [item.to_dict() for item in self.label_hints],
            "metadata": dict(self.metadata),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlateUserHints":
        return cls(
            image=str(payload["image"]),
            image_width=int(payload["image_width"]),
            image_height=int(payload["image_height"]),
            positive_clicks=[PointAnnotation.from_dict(item) for item in payload.get("positive_clicks", [])],
            negative_clicks=[PointAnnotation.from_dict(item) for item in payload.get("negative_clicks", [])],
            label_hints=[PolygonAnnotation.from_dict(item) for item in payload.get("label_hints", [])],
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: Path) -> "PlateUserHints":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def annotation_paths_for_image(root: Path, image_name: str) -> dict[str, Path]:
    stem = Path(image_name).stem
    return {
        "gold": root / "gold" / f"{stem}.gold.json",
        "hints": root / "hints" / f"{stem}.hints.json",
    }
