# Phase 2 Preflight

Date: 2026-05-09

No scoring or pipeline code changes were made in this preflight.

## Organism Metadata

Wrote benchmark plate organism labels to:

```text
data/annotations/benchmark_plate_metadata.json
```

The labels are based on legacy `AGAR_representative` JSON metadata, not visual guessing. Species outside the requested named labels are collapsed only when safe:

- `E.coli` -> `E. coli`
- `S.aureus`, `B.subtilis`, `P.aeruginosa` -> `generic bacteria`
- `C.albicans` -> `unknown`, because it is not `S. cerevisiae` or `K. phaffii`

| Plate | Organism Label | Source Classes | Source Metadata |
| --- | --- | --- | --- |
| 1399 copy.jpg | generic bacteria | S.aureus | `AGAR_representative/higher-resolution/bright/1399.json` |
| 14380 copy.jpg | generic bacteria | B.subtilis | `AGAR_representative/lower-resolution/14380.json` |
| 14410 copy.jpg | generic bacteria | E.coli, S.aureus | `AGAR_representative/lower-resolution/14410.json` |
| 14618 copy.jpg | generic bacteria | S.aureus | `AGAR_representative/lower-resolution/14618.json` |
| 518 copy.jpg | generic bacteria | S.aureus | `AGAR_representative/higher-resolution/bright/518.json` |
| 5207 copy.jpg | E. coli | E.coli | `AGAR_representative/higher-resolution/dark/5207.json` |
| 525 copy.jpg | generic bacteria | B.subtilis | `AGAR_representative/higher-resolution/bright/525.json` |
| 5270 copy.jpg | generic bacteria | B.subtilis | `AGAR_representative/higher-resolution/dark/5270.json` |
| 5271 copy.jpg | E. coli | E.coli | `AGAR_representative/higher-resolution/dark/5271.json` |
| 5308 copy.jpg | generic bacteria | S.aureus | `AGAR_representative/higher-resolution/dark/5308.json` |
| 5312 copy.jpg | E. coli | E.coli | `AGAR_representative/higher-resolution/dark/5312.json` |
| 736 copy.jpg | generic bacteria | P.aeruginosa | `AGAR_representative/higher-resolution/bright/736.json` |
| 971 copy.jpg | unknown | C.albicans | `AGAR_representative/higher-resolution/bright/971.json` |

## 5207 Diagnostic

Generated diagnostic overlay:

```text
reports/5207_diagnostic_overlay.jpg
```

Overlay legend:

- green dish circle: scaled CVAT gold dish usable radius
- blue dish circle: detected pipeline dish usable radius
- yellow circles: gold colony ellipses
- green colony circles: matched predictions
- red colony circles: unmatched predictions
- magenta polygons: CVAT `label_region`

### Metrics

| Measurement | Value |
| --- | ---: |
| True colonies | 6 |
| Predicted colonies | 60 |
| True positives | 5 |
| False positives | 55 |
| False negatives | 1 |
| Label-region false positives | 6 |
| Candidate count | 1487 |
| Label-classified candidate count | 612 |
| Detected colony count | 60 |
| Gold dish center, analysis px | `(733.0, 861.4)` |
| Detected dish center, analysis px | `(729, 862)` |
| Gold dish radius, analysis px | `650.6` |
| Detected dish radius, analysis px | `586.0` |
| Signed radius error | `-64.6` |
| Center error | `4.1` |

### Root Cause

`5207` should not be grouped simply as normal blob over-splitting.

The dish detector is centered correctly but underestimates the dish radius by about 65 analysis pixels. That smaller dish mask likely contributes to the one missed true colony near the lower rim. It does not explain the 55 false positives, because all 60 predictions are still inside the gold dish region and none are near the gold rim band.

The main overcount is interior false-positive scoring on a sparse dark-background plate:

- 60/60 predictions are inside the gold dish.
- 0/60 predictions are outside the gold dish.
- 0/60 predictions are near the gold rim band.
- 6/60 predictions fall inside annotated label regions.
- The unmatched predictions include broad low-chroma gray artifacts, label dots, scratches, and diffuse background marks.
- Median predicted equivalent radius is `14.8` px; mean is `18.3` px, so the error is not just tiny speck noise.
- Median predicted `colony_score` is `0.716`, just above the `0.65` threshold.
- Median predicted warmth score is only `0.127`, meaning many accepted predictions are not warm/tan colonies.
- Median predicted local contrast is `127.8`, so the existing contrast gate treats these artifacts as legitimate colony evidence.

Conclusion: `5207` has two separate issues. Dish radius under-detection causes a rim true-colony miss, while the 60-vs-6 count explosion comes from candidate/scoring accepting interior artifacts as diffuse colonies. It should be tracked as `sparse dark plate artifact FP + dish under-radius`, not folded into the same bucket as true colony blob over-splitting.

### Implications For Pale-Colony Work

Pale-colony support should be guarded against `5207`-style artifacts. A pale score that rewards brightness/neutrality alone could make this plate worse unless it also requires blob coherence, size sanity, and/or a local contrast pattern that separates real smooth colonies from gray label dots and background marks.
