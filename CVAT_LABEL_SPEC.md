# GxP Bio CVAT Label Spec

Date: 2026-04-26

## CVAT Project Name

`GxP Bio Plates`

## Labels

### 1. `dish`

- Shape: `ellipse`
- Required: yes, exactly 1 per plate when visible
- Meaning:
  - outer usable plate region for colony analysis

Store:

- center x/y
- radius x/y
- rotation

### 2. `colony`

- Shape: `points` for the fastest workflow
- Optional upgrade: `ellipse` if we later want explicit size per colony
- Required: yes, one point per visible colony
- Meaning:
  - true colony center

Guidance:

- click once at the visual center of each colony
- for merged/touching colonies, annotate each visible colony center separately if distinguishable

### 3. `label_region`

- Shape: `polygon`
- Required: yes if printed label / writing exists
- Meaning:
  - any non-colony printed or written marking inside the dish

Examples:

- dot-matrix text
- pen marks
- stamp-like text

### 4. `ignore_region`

- Shape: `polygon`
- Optional
- Meaning:
  - region where the pipeline should not be judged harshly

Examples:

- glare
- scratches
- heavy rim artifact
- blur
- contamination streaks that are not part of the colony-counting task

## Annotation Rules

1. Do not annotate plate rim lines as label or colony.
2. Annotate colonies inside the usable dish area only.
3. If a region is ambiguous, prefer putting it in `ignore_region` rather than forcing a wrong colony/non-colony decision.
4. `label_region` should cover the whole printed text cluster, not just a few dots.

## Recommended CVAT Export

Primary export:

- CVAT native image format

Why:

- supports ellipses, points, and polygons directly

Import script:

- `python3 scripts/import_cvat_native.py path/to/annotations.xml --output-dir data/annotations/gold`
