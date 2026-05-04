from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _density_bucket(true_count: int) -> str:
    if true_count < 50:
        return "sparse"
    if true_count < 150:
        return "medium"
    if true_count < 300:
        return "dense"
    return "very_dense"


def _dominant_morphology(record: dict[str, Any]) -> str:
    point_count = int(record.get("true_point_count", 0))
    ellipse_count = int(record.get("true_ellipse_count", 0))
    if point_count and ellipse_count:
        return "mixed"
    if ellipse_count:
        return "ellipse"
    return "point"


def _mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(record.get(key, 0.0)) for record in records) / len(records)


def _group_rows(records: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[group_key])].append(record)

    rows: list[dict[str, Any]] = []
    for name, items in sorted(groups.items()):
        true_positive = sum(int(item["true_positive"]) for item in items)
        false_positive = sum(int(item["false_positive"]) for item in items)
        false_negative = sum(int(item["false_negative"]) for item in items)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)
        rows.append(
            {
                "group": name,
                "images": len(items),
                "true_count": sum(int(item["true_count"]) for item in items),
                "predicted_count": sum(int(item["predicted_count"]) for item in items),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_absolute_count_error": _mean(items, "absolute_count_error"),
                "label_region_false_positives": sum(
                    int(item["label_region_false_positives"]) for item in items
                ),
            }
        )
    return rows


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(evaluation: dict[str, Any]) -> str:
    summary = evaluation["summary"]
    images = list(evaluation["images"])
    for record in images:
        record["density_bucket"] = _density_bucket(int(record["true_count"]))
        record["dominant_morphology"] = _dominant_morphology(record)

    worst_f1 = sorted(images, key=lambda item: (float(item["f1"]), -int(item["true_count"])))[:5]
    worst_count = sorted(images, key=lambda item: int(item["absolute_count_error"]), reverse=True)[:5]
    worst_labels = sorted(images, key=lambda item: int(item["label_region_false_positives"]), reverse=True)[:5]

    density_rows = _group_rows(images, "density_bucket")
    morphology_rows = _group_rows(images, "dominant_morphology")

    lines = [
        "# GxP Gold Benchmark Report",
        "",
        "## Summary",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["Images", str(summary["images"])],
                ["True colonies", str(summary["true_colonies"])],
                ["Point colonies", str(summary["true_point_colonies"])],
                ["Ellipse colonies", str(summary["true_ellipse_colonies"])],
                ["Predicted colonies", str(summary["predicted_colonies"])],
                ["Micro precision", _percent(float(summary["micro_precision"]))],
                ["Micro recall", _percent(float(summary["micro_recall"]))],
                ["Micro F1", _percent(float(summary["micro_f1"]))],
                ["Point recall", _percent(float(summary["point_recall"]))],
                ["Ellipse recall", _percent(float(summary["ellipse_recall"]))],
                ["Mean absolute count error", f"{float(summary['mean_absolute_count_error']):.1f}"],
                ["Label-region false positives", str(summary["total_label_region_false_positives"])],
            ],
        ),
        "",
        "## By Density",
        "",
        _markdown_table(
            ["Bucket", "Images", "True", "Pred", "Precision", "Recall", "F1", "MAE", "Label FP"],
            [
                [
                    row["group"],
                    str(row["images"]),
                    str(row["true_count"]),
                    str(row["predicted_count"]),
                    _percent(row["precision"]),
                    _percent(row["recall"]),
                    _percent(row["f1"]),
                    f"{row['mean_absolute_count_error']:.1f}",
                    str(row["label_region_false_positives"]),
                ]
                for row in density_rows
            ],
        ),
        "",
        "## By Morphology",
        "",
        _markdown_table(
            ["Morphology", "Images", "True", "Pred", "Precision", "Recall", "F1", "MAE", "Label FP"],
            [
                [
                    row["group"],
                    str(row["images"]),
                    str(row["true_count"]),
                    str(row["predicted_count"]),
                    _percent(row["precision"]),
                    _percent(row["recall"]),
                    _percent(row["f1"]),
                    f"{row['mean_absolute_count_error']:.1f}",
                    str(row["label_region_false_positives"]),
                ]
                for row in morphology_rows
            ],
        ),
        "",
        "## Worst F1 Plates",
        "",
        _markdown_table(
            ["Image", "True", "Pred", "F1", "Point Recall", "Ellipse Recall", "Label FP"],
            [
                [
                    row["image"],
                    str(row["true_count"]),
                    str(row["predicted_count"]),
                    _percent(float(row["f1"])),
                    _percent(float(row["point_recall"])),
                    _percent(float(row["ellipse_recall"])),
                    str(row["label_region_false_positives"]),
                ]
                for row in worst_f1
            ],
        ),
        "",
        "## Worst Count Error Plates",
        "",
        _markdown_table(
            ["Image", "True", "Pred", "Error", "Abs Error", "F1"],
            [
                [
                    row["image"],
                    str(row["true_count"]),
                    str(row["predicted_count"]),
                    str(row["count_error"]),
                    str(row["absolute_count_error"]),
                    _percent(float(row["f1"])),
                ]
                for row in worst_count
            ],
        ),
        "",
        "## Worst Label-Region False Positives",
        "",
        _markdown_table(
            ["Image", "Label FP", "True", "Pred", "Precision", "Recall"],
            [
                [
                    row["image"],
                    str(row["label_region_false_positives"]),
                    str(row["true_count"]),
                    str(row["predicted_count"]),
                    _percent(float(row["precision"])),
                    _percent(float(row["recall"])),
                ]
                for row in worst_labels
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def write_csv(evaluation: dict[str, Any], path: Path) -> None:
    rows = list(evaluation["images"])
    if not rows:
        return
    fieldnames = [
        "image",
        "true_count",
        "true_point_count",
        "true_ellipse_count",
        "predicted_count",
        "count_error",
        "absolute_count_error",
        "candidate_count",
        "label_count",
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1",
        "point_recall",
        "ellipse_recall",
        "label_region_false_positives",
        "high_density",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a human-readable report from a GxP gold evaluation JSON.")
    parser.add_argument("evaluation_json", type=Path)
    parser.add_argument("--markdown-path", type=Path, default=Path("outputs/evaluation/gold_benchmark_report.md"))
    parser.add_argument("--csv-path", type=Path, default=Path("outputs/evaluation/gold_benchmark_per_plate.csv"))
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation_json.read_text(encoding="utf-8"))
    report = build_report(evaluation)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.write_text(report, encoding="utf-8")
    write_csv(evaluation, args.csv_path)
    print(f"Wrote {args.markdown_path}")
    print(f"Wrote {args.csv_path}")


if __name__ == "__main__":
    main()
