# Apricot Colony Counter Lab-Grade Roadmap

Date: 2026-05-04

This project should improve like a measurement instrument, not like a one-off demo. The loop is:

1. collect standardized plate images
2. annotate a trusted gold benchmark in CVAT
3. evaluate the current pipeline
4. export model-ready training data
5. train locally
6. compare against a locked holdout set
7. deploy only when metrics improve for the right reasons

## What You Need

### 1. Source Images

Use original plate images, not screenshots.

Recommended capture rules:

- keep the plate centered
- use consistent distance and lighting
- avoid glare if possible
- include sparse, medium, dense, fuzzy, label-overlap, bright, and dark plates
- keep the raw image filenames stable after annotation

Current local benchmark images live in:

```text
data/benchmark/images/
```

### 2. CVAT Gold Annotations

Use CVAT native image export so points, ellipses, and polygons survive the round trip.

Project labels:

- `dish`: ellipse around the usable dish region
- `colony_point`: point for each small dot-like colony
- `colony_ellipse`: ellipse for each large diffuse/blob-like colony
- `label_region`: polygon around printed text or writing
- `ignore_region`: optional polygon for glare, scratches, or ambiguous regions

Import a CVAT native XML export:

```bash
python3 scripts/import_cvat_native.py "/path/to/annotations.xml" --output-dir data/annotations/gold
```

Current imported gold annotations live in:

```text
data/annotations/gold/
```

For each new batch, use a dry-run first:

```bash
python3 scripts/import_cvat_native.py "/path/to/Batch 002.zip" \
  --output-dir data/annotations/gold \
  --image-dir data/benchmark/images \
  --dry-run \
  --no-overwrite \
  --require-images
```

This should list the new image names and zero existing gold files. If it lists the
original 13 benchmark images or 13 existing outputs, the file is probably a project
export rather than the intended task export. Re-export the specific CVAT task in
native image format before importing.

### 3. Benchmark Report

Run the current OpenCV pipeline against the gold annotations:

```bash
python3 scripts/evaluate_gold_annotations.py \
  --image-dir data/benchmark/images \
  --annotations-dir data/annotations/gold \
  --summary-path outputs/evaluation/gold_annotations_current.json
```

Create a readable report:

```bash
python3 scripts/report_gold_benchmark.py \
  outputs/evaluation/gold_annotations_current.json \
  --markdown-path outputs/evaluation/gold_benchmark_report.md \
  --csv-path outputs/evaluation/gold_benchmark_per_plate.csv
```

Watch these metrics first:

- micro precision
- micro recall
- micro F1
- point recall
- ellipse recall
- mean absolute count error
- label-region false positives
- worst plates by F1
- worst plates by count error

### 4. Local Detector Dataset

Export the same CVAT gold data into a YOLO-style local detector dataset:

```bash
python3 scripts/export_detector_dataset.py \
  --image-dir data/benchmark/images \
  --annotations-dir data/annotations/gold \
  --output-dir data/model_training/yolo_gold
```

This writes:

```text
data/model_training/yolo_gold/
  classes.txt
  dataset.yaml
  manifest.json
  images/train/
  images/val/
  labels/train/
  labels/val/
```

The classes are:

- `colony_point`
- `colony_ellipse`
- `label_region`

This does not require Roboflow. It is just local files generated from your own annotations.

## Near-Term Model Strategy

Keep OpenCV for:

- dish detection
- downscaling
- illumination normalization
- rim masking
- debug overlays
- explainable fallback behavior

Add a local supervised detector for:

- point colony proposals
- ellipse/blob colony proposals
- printed label region detection

Then merge model proposals with the current scoring layer instead of replacing the whole app at once.

## Promotion Rules

Do not deploy a detector because one plate looks better.

Deploy when it improves the locked benchmark by:

- lower mean absolute count error
- higher micro F1
- no regression on label-region false positives
- no major regression on either point recall or ellipse recall
- better behavior on the worst five plates

## Current Baseline Reality

The current OpenCV system is useful as a transparent baseline and UI test surface. It is not lab-grade yet.

The next serious milestone is not a prettier overlay. It is a reproducible benchmark report plus a local detector trained from the CVAT gold set.
