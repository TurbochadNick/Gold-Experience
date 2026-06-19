from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gold_experience.gold_evaluation import evaluate_gold_dataset  # noqa: E402
from gold_experience.pipeline import ColonyCounterPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Apricot Colony Counter against CVAT gold annotations.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/benchmark/images"),
        help="Directory containing images named by the *.gold.json files.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/annotations/gold"),
        help="Directory containing Apricot *.gold.json annotation files.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional path to write the full JSON evaluation report.",
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=18.0,
        help="Prediction-to-annotation matching tolerance in analysis-image pixels.",
    )
    args = parser.parse_args()

    pipeline = ColonyCounterPipeline()
    report = evaluate_gold_dataset(
        image_dir=args.image_dir,
        annotations_dir=args.annotations_dir,
        pipeline=pipeline,
        match_tolerance=args.match_tolerance,
    )
    payload = json.dumps(report, indent=2)
    print(payload)

    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
