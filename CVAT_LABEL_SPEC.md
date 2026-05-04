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

### 2. `colony_point`

- Shape: `points`
- Required: yes, one point per visible colony
- Meaning:
  - true center of a small dot-like colony

Guidance:

- use this for dense plates like `1399`
- click once at the visual center of each small colony
- for merged/touching colonies, annotate each visible colony center separately if distinguishable

Legacy compatibility:

- Existing `colony` point labels still import as `colony_point`.

### 3. `colony_ellipse`

- Shape: `ellipse`
- Required: yes for larger diffuse/blob-like colonies when present
- Meaning:
  - approximate footprint and center of a large fuzzy colony

Guidance:

- use this for plates like `525`
- draw a loose ellipse around the visible colony body
- do not trace every fuzzy edge perfectly; center and approximate radius matter most

Legacy compatibility:

- Existing `colony` ellipse labels import as `colony_ellipse`.

### 4. `label_region`

- Shape: `polygon`
- Required: yes if printed label / writing exists
- Meaning:
  - any non-colony printed or written marking inside the dish

Examples:

- dot-matrix text
- pen marks
- stamp-like text

### 5. `ignore_region`

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
5. Use `colony_point` for small dot colonies and `colony_ellipse` for large diffuse colonies.

## Recommended CVAT Export

Primary export:

- CVAT native image format

Why:

- supports ellipses, points, and polygons directly

Import script:

- `python3 scripts/import_cvat_native.py path/to/annotations.xml --output-dir data/annotations/gold`
