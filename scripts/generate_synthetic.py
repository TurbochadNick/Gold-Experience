from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apricot.synthetic import (
    DEFAULT_COLLISION_MARGIN,
    DEFAULT_IMAGE_SIZE,
    SCHEMA_CLEAN_DOTS,
    SCHEMA_MERGED_SNOWMAN,
    generate_dataset,
    generate_synthetic_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Apricot synthetic YOLO data.")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_apricot_v3"))
    parser.add_argument("--plates", type=int, default=100)
    parser.add_argument("--train-plates", type=int, default=None)
    parser.add_argument("--val-plates", type=int, default=None)
    parser.add_argument("--stress-plates", type=int, default=20, help="Images per synthetic stress split.")
    parser.add_argument("--img-size", "--image-size", dest="img_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--species", default="generic_yeast")
    parser.add_argument("--medium", default="generic_dark_agar")
    parser.add_argument(
        "--schema",
        choices=[SCHEMA_CLEAN_DOTS, SCHEMA_MERGED_SNOWMAN],
        default=SCHEMA_CLEAN_DOTS,
        help="Synthetic schema to generate. The named suite remains clean_dots-only.",
    )
    parser.add_argument("--selection-marker", action="append", default=[], help="Freeform antibiotic or marker label.")
    parser.add_argument("--dish-radius", type=int, default=None)
    parser.add_argument("--collision-margin", type=int, default=DEFAULT_COLLISION_MARGIN)
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Write the named robustness suite instead of the default images/train + images/val dataset.",
    )
    parser.add_argument(
        "--size-mode",
        choices=["starter", "small", "medium", "large", "mixed"],
        default="starter",
        help="starter balances small/medium/large/mixed plates.",
    )
    parser.add_argument(
        "--agar-color",
        default=None,
        help="Optional hex override for the medium profile agar color.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.suite:
        if args.schema != SCHEMA_CLEAN_DOTS:
            parser.error("--suite currently writes clean_dots only; omit --suite to generate merged_snowman data.")
        train_plates = args.train_plates
        val_plates = args.val_plates
        if train_plates is None or val_plates is None:
            computed_train = int(round(args.plates * args.train_ratio))
            computed_train = max(1, min(computed_train, args.plates - 1))
            train_plates = computed_train if train_plates is None else train_plates
            val_plates = args.plates - computed_train if val_plates is None else val_plates
        summary = generate_synthetic_suite(
            output_dir=args.out,
            train_plates=train_plates,
            val_plates=val_plates,
            stress_plates=args.stress_plates,
            image_size=args.img_size,
            seed=args.seed,
            species=args.species,
            medium=args.medium,
            dish_radius=args.dish_radius,
            collision_margin=args.collision_margin,
            selection_markers=args.selection_marker,
            overwrite=args.overwrite,
        )
    else:
        summary = generate_dataset(
            output_dir=args.out,
            plates=args.plates,
            image_size=args.img_size,
            train_ratio=args.train_ratio,
            seed=args.seed,
            size_mode=args.size_mode,
            schema=args.schema,
            species=args.species,
            medium=args.medium,
            agar_hex=args.agar_color,
            dish_radius=args.dish_radius,
            collision_margin=args.collision_margin,
            selection_markers=args.selection_marker,
            overwrite=args.overwrite,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
