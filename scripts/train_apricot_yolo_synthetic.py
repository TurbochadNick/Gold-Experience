from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    red, green, blue = (int(hex_color[index : index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


@dataclass(frozen=True)
class Colony:
    x: int
    y: int
    radius: int


SIZE_RANGES: dict[str, tuple[int, int]] = {
    "small": (15, 25),
    "medium": (25, 40),
    "large": (40, 60),
    "mixed": (15, 60),
}


DEFAULT_DATASET_CONFIG: tuple[tuple[int, str, tuple[int, int]], ...] = (
    (25, "small", (60, 120)),
    (25, "medium", (50, 100)),
    (25, "large", (40, 80)),
    (25, "mixed", (50, 120)),
)


def check_collision(
    x_pos: int,
    y_pos: int,
    radius: int,
    existing_colonies: list[Colony],
    min_distance: int = 5,
) -> bool:
    for colony in existing_colonies:
        distance = float(np.hypot(x_pos - colony.x, y_pos - colony.y))
        if distance < radius + colony.radius + min_distance:
            return True
    return False


def _draw_gradient_colony(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    colony_center_bgr: tuple[int, int, int],
    colony_edge_bgr: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = image.copy()
    gradient_steps = 20
    for step in range(gradient_steps):
        t_value = step / gradient_steps
        current_radius = int(radius * (1.0 - t_value))
        color = tuple(
            int(colony_edge_bgr[channel] * t_value + colony_center_bgr[channel] * (1.0 - t_value))
            for channel in range(3)
        )
        if current_radius > 0:
            cv2.circle(overlay, center, current_radius, color, -1, lineType=cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)

    roi_size = radius + 10
    x0 = max(0, center[0] - roi_size)
    y0 = max(0, center[1] - roi_size)
    x1 = min(image.shape[1], center[0] + roi_size)
    y1 = min(image.shape[0], center[1] + roi_size)
    roi = image[y0:y1, x0:x1]
    if roi.size:
        image[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (5, 5), 0)


def generate_plate(
    output_path: Path,
    rng: np.random.Generator,
    num_colonies: int = 50,
    size_mode: str = "mixed",
    image_size: int = 2000,
) -> dict[str, object]:
    if size_mode not in SIZE_RANGES:
        raise ValueError(f"Unknown size mode: {size_mode}")

    dish_center = (image_size // 2, image_size // 2)
    dish_radius = int(image_size * 0.45)
    agar_color = hex_to_bgr("#3A3420")
    dish_outline = hex_to_bgr("#FFFFFF")
    colony_center = hex_to_bgr("#E4DCC6")
    colony_edge = hex_to_bgr("#9C744A")

    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    cv2.circle(image, dish_center, dish_radius, agar_color, -1, lineType=cv2.LINE_AA)

    dish_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.circle(dish_mask, dish_center, dish_radius, 255, -1, lineType=cv2.LINE_AA)
    noise = rng.integers(-15, 15, size=(image_size, image_size, 3), dtype=np.int16)
    image_i16 = image.astype(np.int16)
    image_i16[dish_mask > 0] += noise[dish_mask > 0]
    image = np.clip(image_i16, 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (7, 7), 0)

    cv2.circle(image, dish_center, dish_radius, dish_outline, 3, lineType=cv2.LINE_AA)

    min_radius, max_radius = SIZE_RANGES[size_mode]
    existing_colonies: list[Colony] = []
    annotations: list[str] = []
    max_attempts = num_colonies * 50

    for _ in range(max_attempts):
        if len(existing_colonies) >= num_colonies:
            break

        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        distance = float(rng.uniform(0.0, dish_radius - 150))
        x_pos = int(dish_center[0] + distance * np.cos(angle))
        y_pos = int(dish_center[1] + distance * np.sin(angle))
        radius = int(rng.integers(min_radius, max_radius))

        if check_collision(x_pos, y_pos, radius, existing_colonies):
            continue

        existing_colonies.append(Colony(x=x_pos, y=y_pos, radius=radius))
        _draw_gradient_colony(
            image=image,
            center=(x_pos, y_pos),
            radius=radius,
            colony_center_bgr=colony_center,
            colony_edge_bgr=colony_edge,
            alpha=float(rng.uniform(0.85, 0.95)),
        )
        annotations.append(
            "0 "
            f"{x_pos / image_size:.6f} "
            f"{y_pos / image_size:.6f} "
            f"{(radius * 2) / image_size:.6f} "
            f"{(radius * 2) / image_size:.6f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    output_path.with_suffix(".txt").write_text("\n".join(annotations) + "\n", encoding="utf-8")

    return {
        "image": output_path.name,
        "size_mode": size_mode,
        "requested_colonies": num_colonies,
        "placed_colonies": len(existing_colonies),
        "image_size": image_size,
        "dish": {
            "x": dish_center[0],
            "y": dish_center[1],
            "radius": dish_radius,
        },
        "colors": {
            "agar_hex": "#3A3420",
            "dish_outline_hex": "#FFFFFF",
            "colony_center_hex": "#E4DCC6",
            "colony_edge_hex": "#9C744A",
        },
    }


def generate_dataset(
    output_dir: Path,
    dataset_root: Path,
    seed: int,
    image_size: int,
    train_ratio: float,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    plate_index = 0

    for count, size_mode, colony_range in DEFAULT_DATASET_CONFIG:
        min_colonies, max_colonies = colony_range
        for _ in range(count):
            requested_colonies = int(rng.integers(min_colonies, max_colonies + 1))
            plate_path = output_dir / f"plate_{plate_index:04d}.jpg"
            record = generate_plate(
                output_path=plate_path,
                rng=rng,
                num_colonies=requested_colonies,
                size_mode=size_mode,
                image_size=image_size,
            )
            manifest.append(record)
            plate_index += 1

    image_paths = sorted(output_dir.glob("plate_*.jpg"))
    train_count = int(round(len(image_paths) * train_ratio))
    train_count = max(1, min(train_count, len(image_paths)))

    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(image_paths):
        split = "train" if index < train_count else "val"
        shutil.copy2(image_path, dataset_root / "images" / split / image_path.name)
        shutil.copy2(image_path.with_suffix(".txt"), dataset_root / "labels" / split / image_path.with_suffix(".txt").name)

    dataset_yaml = "\n".join(
        [
            f"path: {dataset_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "nc: 1",
            "names:",
            "  0: yeast_colony",
            "",
        ]
    )
    (dataset_root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    (dataset_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "plates": len(image_paths),
        "train": train_count,
        "val": len(image_paths) - train_count,
        "total_colonies": sum(int(record["placed_colonies"]) for record in manifest),
        "synthetic_images": str(output_dir),
        "dataset": str(dataset_root),
    }


def train_model(dataset_root: Path, epochs: int, image_size: int, batch: int, run_name: str) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Install YOLO tooling in a local training environment "
            "that preserves opencv-python-headless, then rerun with --train."
        ) from exc

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(dataset_root / "dataset.yaml"),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        patience=10,
        name=run_name,
        lr0=0.001,
        lrf=0.001,
        weight_decay=0.0005,
        val=True,
        plots=True,
        save=True,
        save_period=10,
    )

    best_weights = Path("runs") / "detect" / run_name / "weights" / "best.pt"
    best_model = YOLO(str(best_weights))
    val_results = best_model.val()
    print("\nFinal model metrics")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"Precision: {val_results.box.p[0]:.4f}")
    print(f"Recall: {val_results.box.r[0]:.4f}")
    print(f"Model weights: {best_weights}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Apricot synthetic YOLO data and optionally train YOLOv8.")
    parser.add_argument("--synthetic-dir", type=Path, default=Path("data/generated/apricot_synthetic_v3"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/model_training/apricot_yolo_synthetic_v3"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--image-size", type=int, default=2000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--train", action="store_true", help="Train YOLOv8 after generating the dataset.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--run-name", default="apricot_v3_realistic")
    args = parser.parse_args()

    summary = generate_dataset(
        output_dir=args.synthetic_dir,
        dataset_root=args.dataset_root,
        seed=args.seed,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
    )
    print(json.dumps(summary, indent=2))
    print(f"Dataset YAML: {args.dataset_root / 'dataset.yaml'}")

    if args.train:
        train_model(
            dataset_root=args.dataset_root,
            epochs=args.epochs,
            image_size=640,
            batch=args.batch,
            run_name=args.run_name,
        )


if __name__ == "__main__":
    main()
