from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gold_experience.synthetic import generate_plate, save_plate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Gold Experience plates.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--label-text", type=str, default=None)
    parser.add_argument("--min-colonies", type=int, default=18)
    parser.add_argument("--max-colonies", type=int, default=45)
    args = parser.parse_args()

    for index in range(args.count):
        plate = generate_plate(
            width=args.width,
            height=args.height,
            colony_count_range=(args.min_colonies, args.max_colonies),
            label_text=args.label_text,
            seed=None if args.seed is None else args.seed + index,
        )
        stem = f"plate_{index:04d}"
        save_plate(plate, args.output_dir, stem)
        print(
            f"{stem}: colonies={plate.metadata['colony_count']} "
            f"label_dots={len(plate.metadata['label_dots'])}"
        )


if __name__ == "__main__":
    main()

