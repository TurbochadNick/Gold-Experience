from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gold_experience.annotations import PlateGoldAnnotation, PolygonAnnotation  # noqa: E402


CLASSES = ["colony_point", "colony_ellipse", "label_region"]


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _box_from_center(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    class_id: int,
) -> YoloBox | None:
    x_min = _clamp(x_center - width / 2.0, 0.0, float(image_width))
    y_min = _clamp(y_center - height / 2.0, 0.0, float(image_height))
    x_max = _clamp(x_center + width / 2.0, 0.0, float(image_width))
    y_max = _clamp(y_center + height / 2.0, 0.0, float(image_height))
    clipped_width = x_max - x_min
    clipped_height = y_max - y_min
    if clipped_width <= 1.0 or clipped_height <= 1.0:
        return None
    return YoloBox(
        class_id=class_id,
        x_center=((x_min + x_max) / 2.0) / image_width,
        y_center=((y_min + y_max) / 2.0) / image_height,
        width=clipped_width / image_width,
        height=clipped_height / image_height,
    )


def _polygon_bbox(polygon: PolygonAnnotation) -> tuple[float, float, float, float] | None:
    if not polygon.points:
        return None
    xs = [point[0] for point in polygon.points]
    ys = [point[1] for point in polygon.points]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def _boxes_for_annotation(gold: PlateGoldAnnotation, point_box_radius: float) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    for colony in gold.colonies:
        morphology = colony.morphology.lower()
        if morphology == "ellipse":
            rx = colony.rx if colony.rx is not None else colony.radius
            ry = colony.ry if colony.ry is not None else colony.radius
            if rx is None or ry is None:
                rx = ry = point_box_radius
            box = _box_from_center(
                x_center=colony.x,
                y_center=colony.y,
                width=max(2.0, 2.0 * float(rx)),
                height=max(2.0, 2.0 * float(ry)),
                image_width=gold.image_width,
                image_height=gold.image_height,
                class_id=1,
            )
        else:
            radius = colony.radius if colony.radius is not None else point_box_radius
            box = _box_from_center(
                x_center=colony.x,
                y_center=colony.y,
                width=max(2.0, 2.0 * float(radius)),
                height=max(2.0, 2.0 * float(radius)),
                image_width=gold.image_width,
                image_height=gold.image_height,
                class_id=0,
            )
        if box is not None:
            boxes.append(box)

    for label_region in gold.label_regions:
        bbox = _polygon_bbox(label_region)
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        box = _box_from_center(
            x_center=(x_min + x_max) / 2.0,
            y_center=(y_min + y_max) / 2.0,
            width=x_max - x_min,
            height=y_max - y_min,
            image_width=gold.image_width,
            image_height=gold.image_height,
            class_id=2,
        )
        if box is not None:
            boxes.append(box)
    return boxes


def _write_yolo_label(path: Path, boxes: Iterable[YoloBox]) -> None:
    lines = [
        f"{box.class_id} {box.x_center:.8f} {box.y_center:.8f} {box.width:.8f} {box.height:.8f}"
        for box in boxes
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _safe_stem(image_name: str) -> str:
    return Path(image_name).stem.replace(" ", "_")


def export_dataset(
    image_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    train_ratio: float,
    seed: int,
    point_box_radius: float,
) -> dict[str, object]:
    annotation_paths = sorted(annotations_dir.glob("*.gold.json"))
    records: list[tuple[PlateGoldAnnotation, Path, list[YoloBox]]] = []
    missing_images: list[str] = []

    for annotation_path in annotation_paths:
        gold = PlateGoldAnnotation.load(annotation_path)
        image_path = image_dir / gold.image
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue
        records.append((gold, image_path, _boxes_for_annotation(gold, point_box_radius)))

    rng = random.Random(seed)
    rng.shuffle(records)
    train_count = int(round(len(records) * train_ratio))
    train_count = max(1, min(len(records), train_count)) if records else 0
    split_records = {
        "train": records[:train_count],
        "val": records[train_count:],
    }
    if records and not split_records["val"]:
        split_records["val"] = split_records["train"][-1:]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "classes": CLASSES,
        "source_image_dir": str(image_dir),
        "source_annotations_dir": str(annotations_dir),
        "missing_images": missing_images,
        "splits": {},
    }

    for split, items in split_records.items():
        split_manifest: list[dict[str, object]] = []
        for gold, image_path, boxes in items:
            image_destination = output_dir / "images" / split / image_path.name
            label_destination = output_dir / "labels" / split / f"{_safe_stem(gold.image)}.txt"
            _copy_image(image_path, image_destination)
            _write_yolo_label(label_destination, boxes)
            split_manifest.append(
                {
                    "image": gold.image,
                    "image_path": str(image_destination),
                    "label_path": str(label_destination),
                    "colonies": len(gold.colonies),
                    "point_colonies": sum(1 for colony in gold.colonies if colony.morphology == "point"),
                    "ellipse_colonies": sum(1 for colony in gold.colonies if colony.morphology == "ellipse"),
                    "label_regions": len(gold.label_regions),
                    "boxes": len(boxes),
                }
            )
        manifest["splits"][split] = split_manifest  # type: ignore[index]

    (output_dir / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    (output_dir / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(CLASSES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Apricot gold annotations into a local YOLO-style detector dataset."
    )
    parser.add_argument("--image-dir", type=Path, default=Path("data/benchmark/images"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/annotations/gold"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/model_training/yolo_gold"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--point-box-radius",
        type=float,
        default=8.0,
        help="Half-size in pixels for point colony training boxes.",
    )
    args = parser.parse_args()

    manifest = export_dataset(
        image_dir=args.image_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        point_box_radius=args.point_box_radius,
    )
    splits = manifest["splits"]
    train_items = len(splits.get("train", []))  # type: ignore[union-attr]
    val_items = len(splits.get("val", []))  # type: ignore[union-attr]
    print(f"Wrote detector dataset to {args.output_dir}")
    print(f"Classes: {', '.join(CLASSES)}")
    print(f"Images: {train_items} train, {val_items} val")
    if manifest["missing_images"]:
        print(f"Missing images: {len(manifest['missing_images'])}")


if __name__ == "__main__":
    main()
