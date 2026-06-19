from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apricot.inference import DEFAULT_IOU, DEFAULT_MAX_DET, DEFAULT_MODEL_PATH, predict_image_array

DEFAULT_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_manifest_path(suite_dir: Path, manifest_value: str, split_name: str) -> Path:
    manifest_path = Path(manifest_value)
    if manifest_path.is_file():
        return manifest_path
    if not manifest_path.is_absolute():
        candidate = (suite_dir / manifest_path).resolve()
        if candidate.is_file():
            return candidate
    return suite_dir / split_name / "manifest.json"


def iter_split_manifests(suite_dir: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    suite_manifest_path = suite_dir / "suite_manifest.json"
    split_items: list[tuple[str, Path, dict[str, Any]]] = []

    if suite_manifest_path.is_file():
        suite_manifest = _load_json(suite_manifest_path)
        for split_summary in suite_manifest.get("splits", []):
            split_name = str(split_summary["name"])
            manifest_path = _resolve_manifest_path(suite_dir, str(split_summary["manifest"]), split_name)
            split_items.append((split_name, manifest_path.parent, _load_json(manifest_path)))
        return split_items

    for manifest_path in sorted(suite_dir.glob("*/manifest.json")):
        manifest = _load_json(manifest_path)
        split_items.append((str(manifest.get("split", manifest_path.parent.name)), manifest_path.parent, manifest))
    return split_items


def count_labels(label_path: Path) -> int:
    if not label_path.is_file():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def expected_count(record: dict[str, Any], split_dir: Path) -> int:
    for key in ("expected_colony_count", "placed_colonies", "label_count"):
        if key in record:
            return int(record[key])
    return count_labels(split_dir / str(record["label"]))


def percent_error(abs_error: int, expected: int, predicted: int) -> float:
    if expected == 0:
        return 0.0 if predicted == 0 else 100.0
    return abs_error / expected * 100.0


def evaluate_suite(
    *,
    suite_dir: Path,
    model_path: Path,
    output_dir: Path,
    thresholds: tuple[float, ...],
    iou: float = DEFAULT_IOU,
    max_det: int = DEFAULT_MAX_DET,
    samples_per_split: int = 3,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_rows: list[dict[str, Any]] = []
    split_manifests = iter_split_manifests(suite_dir)
    if not split_manifests:
        raise SystemExit(f"No split manifests found under {suite_dir}")

    for split_name, split_dir, manifest in split_manifests:
        tier = str(manifest.get("tier", "unknown"))
        records = list(manifest.get("images", []))
        for threshold in thresholds:
            for record in records:
                image_path = split_dir / str(record["image"])
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise SystemExit(f"Could not read image: {image_path}")

                result = predict_image_array(
                    image,
                    confidence=threshold,
                    iou=iou,
                    max_det=max_det,
                    model_path=model_path,
                )
                expected = expected_count(record, split_dir)
                predicted = int(result.count)
                signed_error = predicted - expected
                abs_error = abs(signed_error)
                per_image_rows.append(
                    {
                        "split": split_name,
                        "tier": tier,
                        "threshold": f"{threshold:.2f}",
                        "image": str(image_path.relative_to(suite_dir)),
                        "label": str((split_dir / str(record["label"])).relative_to(suite_dir)),
                        "expected_count": expected,
                        "predicted_count": predicted,
                        "signed_count_error": signed_error,
                        "absolute_count_error": abs_error,
                        "percent_count_error": f"{percent_error(abs_error, expected, predicted):.4f}",
                    }
                )

    summary_rows = build_summary_rows(per_image_rows)
    best_by_split = {
        row["split"]: float(row["threshold"])
        for row in summary_rows
        if row["is_best_threshold"] == "true"
    }
    save_annotated_samples(
        suite_dir=suite_dir,
        output_dir=output_dir / "annotated_samples",
        split_manifests=split_manifests,
        best_by_split=best_by_split,
        per_image_rows=per_image_rows,
        model_path=model_path,
        iou=iou,
        max_det=max_det,
        samples_per_split=samples_per_split,
    )

    per_image_csv = output_dir / "synthetic_robustness_per_image.csv"
    summary_csv = output_dir / "synthetic_robustness_summary.csv"
    write_csv(per_image_csv, per_image_rows)
    write_csv(summary_csv, summary_rows)
    return {"per_image_csv": per_image_csv, "summary_csv": summary_csv}


def build_summary_rows(per_image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_image_rows:
        grouped[(str(row["split"]), str(row["threshold"]))].append(row)

    raw_rows: list[dict[str, Any]] = []
    for (split_name, threshold), rows in grouped.items():
        tier = str(rows[0]["tier"])
        abs_errors = [float(row["absolute_count_error"]) for row in rows]
        pct_errors = [float(row["percent_count_error"]) for row in rows]
        signed_errors = [float(row["signed_count_error"]) for row in rows]
        exact_matches = [1.0 if float(row["absolute_count_error"]) == 0.0 else 0.0 for row in rows]
        raw_rows.append(
            {
                "split": split_name,
                "tier": tier,
                "threshold": threshold,
                "images": len(rows),
                "mean_absolute_error": mean(abs_errors),
                "mean_percent_error": mean(pct_errors),
                "mean_signed_error": mean(signed_errors),
                "max_absolute_error": max(abs_errors),
                "exact_count_rate": mean(exact_matches),
            }
        )

    best_thresholds: dict[str, str] = {}
    for split_name in sorted({row["split"] for row in raw_rows}):
        candidates = [row for row in raw_rows if row["split"] == split_name]
        best = min(
            candidates,
            key=lambda row: (
                float(row["mean_absolute_error"]),
                float(row["mean_percent_error"]),
                abs(float(row["mean_signed_error"])),
                float(row["threshold"]),
            ),
        )
        best_thresholds[split_name] = str(best["threshold"])

    summary_rows: list[dict[str, Any]] = []
    for row in sorted(raw_rows, key=lambda item: (str(item["split"]), float(item["threshold"]))):
        best_threshold = best_thresholds[str(row["split"])]
        summary_rows.append(
            {
                "split": row["split"],
                "tier": row["tier"],
                "threshold": row["threshold"],
                "images": row["images"],
                "mean_absolute_error": f"{float(row['mean_absolute_error']):.4f}",
                "mean_percent_error": f"{float(row['mean_percent_error']):.4f}",
                "mean_signed_error": f"{float(row['mean_signed_error']):.4f}",
                "max_absolute_error": f"{float(row['max_absolute_error']):.4f}",
                "exact_count_rate": f"{float(row['exact_count_rate']):.4f}",
                "best_threshold_for_split": best_threshold,
                "is_best_threshold": "true" if str(row["threshold"]) == best_threshold else "false",
            }
        )
    return summary_rows


def save_annotated_samples(
    *,
    suite_dir: Path,
    output_dir: Path,
    split_manifests: list[tuple[str, Path, dict[str, Any]]],
    best_by_split: dict[str, float],
    per_image_rows: list[dict[str, Any]],
    model_path: Path,
    iou: float,
    max_det: int,
    samples_per_split: int,
) -> None:
    if samples_per_split <= 0:
        return

    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_image_rows:
        best_threshold = best_by_split.get(str(row["split"]))
        if best_threshold is None or float(row["threshold"]) != best_threshold:
            continue
        rows_by_split[str(row["split"])].append(row)

    for split_name, rows in rows_by_split.items():
        sample_rows = sorted(rows, key=lambda row: float(row["absolute_count_error"]), reverse=True)[:samples_per_split]
        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        best_threshold = best_by_split[split_name]

        for row in sample_rows:
            image_path = suite_dir / str(row["image"])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            stem = (
                f"{Path(str(row['image'])).stem}"
                f"_thr_{best_threshold:.2f}"
                f"_exp_{row['expected_count']}"
                f"_pred_{row['predicted_count']}"
            )
            predict_image_array(
                image,
                confidence=best_threshold,
                iou=iou,
                max_det=max_det,
                model_path=model_path,
                output_dir=split_output_dir,
                output_stem=stem,
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_thresholds(values: list[str] | None) -> tuple[float, ...]:
    if not values:
        return DEFAULT_THRESHOLDS
    thresholds: list[float] = []
    for value in values:
        for chunk in value.split(","):
            chunk = chunk.strip()
            if chunk:
                thresholds.append(float(chunk))
    return tuple(thresholds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Apricot YOLO robustness on synthetic-only stress splits.")
    parser.add_argument("--suite", type=Path, default=Path("data/generated/apricot_synthetic_suite_v1"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out", type=Path, default=Path("outputs/synthetic_robustness"))
    parser.add_argument("--thresholds", nargs="*", default=None, help="Space- or comma-separated thresholds.")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument("--samples-per-split", type=int, default=3)
    args = parser.parse_args()

    outputs = evaluate_suite(
        suite_dir=args.suite,
        model_path=args.model,
        output_dir=args.out,
        thresholds=parse_thresholds(args.thresholds),
        iou=args.iou,
        max_det=args.max_det,
        samples_per_split=args.samples_per_split,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
