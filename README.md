# Gold Experience

Gold Experience is a rule-based MVP scaffold for plate detection, label rejection, colony counting, synthetic dataset generation, and evaluation.

## V1 pipeline

1. Detect the dish circle.
2. Detect all blob-like candidates inside the plate.
3. Classify candidates as `label` or `not label`.
4. Score surviving `not label` candidates as plausible colonies.
5. Count and render overlays.

The synthetic generator produces automatic ground truth:

- dish circle
- colony mask
- label mask
- colony centers and radii
- label dot centers and radii

## Layout

```text
src/gold_experience/
  synthetic.py
  plate_detection.py
  candidate_detection.py
  label_filter.py
  colony_scoring.py
  pipeline.py
  visualization.py
  evaluation.py
scripts/
  generate_synthetic.py
  run_pipeline.py
  evaluate_pipeline.py
  evaluate_gold_annotations.py
  export_detector_dataset.py
  report_gold_benchmark.py
  import_cvat_native.py
```

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Generate synthetic plates

```bash
python3 scripts/generate_synthetic.py data/generated/synthetic --count 25 --seed 7
```

## Run the pipeline

```bash
python3 scripts/run_pipeline.py data/generated/synthetic --output-dir outputs/run
```

## Evaluate on synthetic data

```bash
python3 scripts/evaluate_pipeline.py data/generated/synthetic
```

This scaffold is intentionally explainable and easy to modify. It is designed to evolve toward real plate images, not to hide the logic inside a black-box detector.

## Local web app

Run the web app and API together:

```bash
python3 scripts/serve_api.py --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The browser UI uploads the image to the same service. The response includes:

- image size
- detected dish circle
- colony detections
- filtered labels
- rejected candidates
- pipeline step summaries for the UI

The API is still available directly at `POST /analyze` and `GET /health`.

## Annotation Workflow

We now have a first scaffold for manual plate annotation with CVAT plus per-image user hints.

Docs:

- [CVAT label spec](./CVAT_LABEL_SPEC.md)
- [Annotation schema](./ANNOTATION_SCHEMA.md)
- [CVAT + GxP integration plan](./CVAT_GXP_PLAN.md)

Python models:

- `src/gold_experience/annotations.py`

Import a CVAT native export:

```bash
python3 scripts/import_cvat_native.py path/to/annotations.xml --output-dir data/annotations/gold
```

Expected CVAT labels:

- `dish`
- `colony_point` for small dot-like colonies
- `colony_ellipse` for larger diffuse/blob-like colonies
- `label_region`
- `ignore_region`

Legacy `colony` labels are still accepted: point shapes import as point colonies, and ellipse shapes import as ellipse colonies.

Evaluate the current detector against the imported gold annotations:

```bash
python3 scripts/evaluate_gold_annotations.py \
  --image-dir data/benchmark/images \
  --annotations-dir data/annotations/gold \
  --summary-path outputs/evaluation/gold_annotations_current.json
```

Create a human-readable benchmark report:

```bash
python3 scripts/report_gold_benchmark.py \
  outputs/evaluation/gold_annotations_current.json \
  --markdown-path outputs/evaluation/gold_benchmark_report.md \
  --csv-path outputs/evaluation/gold_benchmark_per_plate.csv
```

Export the same gold annotations into a local YOLO-style detector dataset:

```bash
python3 scripts/export_detector_dataset.py \
  --image-dir data/benchmark/images \
  --annotations-dir data/annotations/gold \
  --output-dir data/model_training/yolo_gold
```

This creates local training files with three classes:

- `colony_point`
- `colony_ellipse`
- `label_region`

The gold benchmark intentionally lives in `data/benchmark/images` and `data/annotations/gold` so the same annotated plates can be used locally, in CI, and inside the deployed container when needed.

For the full validation loop, see [LAB_GRADE_ROADMAP.md](./LAB_GRADE_ROADMAP.md).

## Deploy with Docker

Build and run locally:

```bash
docker build -t gold-experience .
docker run --rm -p 8000:8000 gold-experience
```

For a hosted deployment, the repo is now structured as a single container service:

- no OpenAI or Roboflow key required
- one web process serves both the UI and the analysis endpoint
- uploads are processed on the server CPU with OpenCV

That makes it straightforward to deploy on Render, Railway, Fly.io, or a lab-managed VM.

## Notes

- `frontend/GoldExperienceConnectedApp.jsx` is still in the repo as a richer React direction, but the deployable MVP now lives in `web/`.
- This is a productization pass, not a model-quality pass. The next step after deployment is tuning the candidate detector and label gate on real plate images.
