from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apricot.synthetic import generate_dataset

SMOKE_COLONY_RANGES = {
    "small": (6, 12),
    "medium": (6, 12),
    "large": (4, 9),
    "mixed": (6, 12),
}


def generate_smoke_dataset(
    *,
    output_dir: Path,
    plates: int = 12,
    image_size: int = 512,
    seed: int = 17,
    species: str = "s_cerevisiae",
    medium: str = "YPD",
    overwrite: bool = False,
) -> dict[str, object]:
    return generate_dataset(
        output_dir=output_dir,
        plates=plates,
        image_size=image_size,
        train_ratio=0.75,
        seed=seed,
        size_mode="starter",
        species=species,
        medium=medium,
        overwrite=overwrite,
        colony_count_ranges=SMOKE_COLONY_RANGES,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tiny Apricot YOLO dataset for smoke checks.")
    parser.add_argument("--out", type=Path, default=Path("data/generated/apricot_smoke_12"))
    parser.add_argument("--plates", type=int, default=12)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--species", default="s_cerevisiae")
    parser.add_argument("--medium", default="YPD")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = generate_smoke_dataset(
        output_dir=args.out,
        plates=args.plates,
        image_size=args.img_size,
        seed=args.seed,
        species=args.species,
        medium=args.medium,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
