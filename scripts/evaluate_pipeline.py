from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gold_experience.evaluation import evaluate_dataset
from gold_experience.pipeline import ColonyCounterPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Gold Experience pipeline on a dataset.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--summary-path", type=Path, default=None)
    args = parser.parse_args()

    pipeline = ColonyCounterPipeline()
    report = evaluate_dataset(args.dataset_dir, pipeline)
    payload = json.dumps(report, indent=2)
    print(payload)

    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
