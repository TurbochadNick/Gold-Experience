# Apricot Benchmark Forensics v1

Date: 2026-05-08

Source run:

```bash
python3 scripts/evaluate_gold_annotations.py \
  --image-dir data/benchmark/images \
  --annotations-dir data/annotations/gold \
  --summary-path outputs/evaluation/gold_annotations_current.json
```

## Baseline Summary

| Metric | Value |
| --- | ---: |
| Images | 13 |
| True colonies | 1509 |
| Predicted colonies | 1687 |
| True positives | 916 |
| False positives | 771 |
| False negatives | 593 |
| Micro precision | 54.3% |
| Micro recall | 60.7% |
| Micro F1 | 57.3% |
| Point recall | 61.4% |
| Ellipse recall | 57.6% |
| Mean absolute count error | 68.9 |
| Label-region false positives | 69 |

## Per-Plate Breakdown

| Plate | True | Pred | Precision | Recall | F1 | Count Error | Label FP | Dominant Error Mode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1399 copy.jpg | 750 | 535 | 84.3% | 60.1% | 70.2% | -215 | 1 | density undercount |
| 14380 copy.jpg | 34 | 18 | 0.0% | 0.0% | 0.0% | -16 | 4 | pale-colony miss |
| 14410 copy.jpg | 33 | 9 | 0.0% | 0.0% | 0.0% | -24 | 0 | pale-colony miss |
| 14618 copy.jpg | 40 | 79 | 46.8% | 92.5% | 62.2% | 39 | 0 | other false-positive noise |
| 518 copy.jpg | 294 | 236 | 92.8% | 74.5% | 82.6% | -58 | 6 | density undercount |
| 5207 copy.jpg | 6 | 60 | 8.3% | 83.3% | 15.2% | 54 | 6 | blob over-split |
| 525 copy.jpg | 85 | 203 | 31.5% | 75.3% | 44.4% | 118 | 4 | blob over-split |
| 5270 copy.jpg | 34 | 83 | 27.7% | 67.6% | 39.3% | 49 | 13 | label FP |
| 5271 copy.jpg | 41 | 115 | 28.7% | 80.5% | 42.3% | 74 | 4 | blob over-split |
| 5308 copy.jpg | 10 | 11 | 81.8% | 90.0% | 85.7% | 1 | 2 | other |
| 5312 copy.jpg | 25 | 125 | 16.8% | 84.0% | 28.0% | 100 | 7 | blob over-split |
| 736 copy.jpg | 28 | 130 | 13.1% | 60.7% | 21.5% | 102 | 17 | label FP |
| 971 copy.jpg | 129 | 83 | 44.6% | 28.7% | 34.9% | -46 | 5 | density undercount |

## Failure Mode Burden

`FP + FN` is the direct aggregate-F1 burden. It is not the whole product story, but it is a useful first ranking signal.

| Failure Mode | Plates | True | Pred | TP | FP | FN | FP + FN | Label FP | Group F1 | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| density undercount | 3 | 1173 | 854 | 707 | 147 | 466 | 613 | 12 | 69.8% | Mostly high-count point plates. Precision is good, recall is the leak. |
| blob over-split | 4 | 157 | 503 | 123 | 380 | 34 | 414 | 21 | 37.3% | Ellipse/blob plates get many extra detections per true colony. |
| label FP | 2 | 62 | 213 | 40 | 173 | 22 | 195 | 30 | 29.1% | Label/rim-adjacent structure is being counted as colonies. |
| pale-colony miss | 2 | 67 | 27 | 0 | 27 | 67 | 94 | 4 | 0.0% | Catastrophic per-plate: no truth matches on either plate. |
| other false-positive noise | 1 | 40 | 79 | 37 | 42 | 3 | 45 | 0 | 62.2% | Recall is high, but generic FP suppression could help. |
| other | 1 | 10 | 11 | 9 | 2 | 1 | 3 | 2 | 85.7% | Basically acceptable on current metrics. |

## Plate Notes

- `1399 copy.jpg`: very dense point-colony plate. Precision is strong at 84.3%, but 299 false negatives drive a -215 count error. This is a density/recall problem more than a false-positive problem.
- `14380 copy.jpg`: 34 ellipse colonies, zero matches. The detector produces 18 predictions, but none match gold. This is the clearest pale/low-contrast failure.
- `14410 copy.jpg`: same failure class as `14380 copy.jpg`: 33 true colonies, zero matches, only 9 predictions. It likely needs new pale-candidate evidence, not just threshold tuning.
- `14618 copy.jpg`: high recall at 92.5%, but 42 false positives. This looks like a generic precision/noise gate issue rather than label-region leakage.
- `518 copy.jpg`: dense point plate with strong precision and moderate recall. It is a safer target for recall tuning than the noisier sparse/blob plates.
- `5207 copy.jpg`: only 6 true ellipse colonies but 60 predictions. Most of the harm is false positives, with a large dish radius error suggesting rim/background structure may be entering the candidate pool.
- `525 copy.jpg`: ellipse/blob-heavy plate. Recall is decent at 75.3%, but 139 false positives dominate the count error.
- `5270 copy.jpg`: mixed morphology with 13 label-region false positives and a large dish radius error. Treat as label/rim false-positive work before recall work.
- `5271 copy.jpg`: ellipse/blob plate with high recall and 82 false positives. This is over-splitting/noise around blob candidates.
- `5308 copy.jpg`: currently healthy enough to treat as a regression sentinel, especially because dish radius error is large but count quality remains good.
- `5312 copy.jpg`: ellipse/blob plate with 21 of 25 truths matched but 104 false positives. The current scoring finds the plate but dramatically overcounts.
- `736 copy.jpg`: worst label-region FP plate, plus large dish radius error. It should be a primary label/rim regression test.
- `971 copy.jpg`: medium-density point plate with only 28.7% recall. Candidate/label gating may be suppressing true point colonies.

## Fix Ranking By Expected Aggregate F1

1. **Reduce blob over-split false positives on ellipse/blob plates.**
   This group contributes 380 false positives and 414 total FP/FN burden. A conservative gate that removes even 25-35% of these false positives without losing current true positives should move aggregate F1 more than a pale-only change. Regression risks: undercounting true diffuse colonies on `525`, `5271`, and `5312`.

2. **Improve dense point recall without lowering precision.**
   Density undercount contributes the largest total burden: 466 false negatives across `1399`, `518`, and `971`. Because this group already has 82.8% precision, recall improvements here are high-value. Regression risks: turning dense plate noise into false positives and worsening label-region counts.

3. **Tighten label/rim rejection on `736` and `5270`.**
   These two plates account for 30 of 69 label-region false positives and 173 false positives overall. This is less aggregate burden than blob over-split, but it directly protects the explainable label filter and should be mandatory before any precision-lowering recall work.

4. **Add pale-colony evidence and scoring.**
   Pale-colony miss is only 94 FP/FN burden in the current 13-plate benchmark, but it is the most severe per-plate failure: `14380` and `14410` both score 0.0% F1. This should still be prioritized for product coverage, then judged against the full benchmark to ensure tan/dark performance does not regress.

5. **Generic false-positive cleanup on `14618`.**
   `14618` has high recall and 42 false positives with no label-region FPs. It is useful as a precision-tuning sentinel, but lower priority than the repeated blob/label/density patterns.

## Recommendation For Phase 2

Start Phase 2 with pale-colony support because it is a known product-critical blind spot, but benchmark it as a targeted coverage fix rather than expecting it to maximize aggregate F1 on this 13-plate set. The pass/fail bar should be:

- `14380 copy.jpg` and `14410 copy.jpg` stop being zero-match plates.
- Aggregate micro F1 does not fall.
- Label-region false positives do not increase above 69.
- Dense/tan sentinel plates `1399 copy.jpg`, `518 copy.jpg`, and `5308 copy.jpg` do not materially regress.

Then move to blob over-split and label/rim false-positive controls, since those have the larger aggregate-F1 upside.
