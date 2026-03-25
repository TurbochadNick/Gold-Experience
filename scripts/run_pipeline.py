from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gold_experience.pipeline import ColonyCounterPipeline
from gold_experience.visualization import render_overlay


def _iter_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    image_paths: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(sorted(input_path.glob(pattern)))

    return [
        path
        for path in image_paths
        if ".colonies." not in path.name and ".labels." not in path.name and not path.name.endswith(".overlay.png")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gold Experience rule-based pipeline.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    parser.add_argument("--save-debug", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ColonyCounterPipeline()

    for image_path in _iter_image_paths(args.input_path):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        result = pipeline.run(image)
        overlay = render_overlay(image, result)
        overlay_path = args.output_dir / f"{image_path.stem}.overlay.png"
        json_path = args.output_dir / f"{image_path.stem}.pred.json"
        cv2.imwrite(str(overlay_path), overlay)

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2)

        if args.save_debug:
            for name, debug_image in result.debug_images.items():
                debug_path = args.output_dir / f"{image_path.stem}.{name}.png"
                cv2.imwrite(str(debug_path), debug_image)

        print(f"{image_path.name}: colonies={result.colony_count} labels={len(result.label_ids)}")


if __name__ == "__main__":
    main()

