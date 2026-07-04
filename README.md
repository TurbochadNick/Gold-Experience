# Apricot Colony Counter

Apricot is a lightweight yeast colony counting web app. It serves a browser upload UI, accepts one plate image or a small batch, runs YOLO-based colony detection, and returns colony counts, detection boxes, schema routing metadata, and annotated image previews.

The production app is deliberately only an inference service. Synthetic data generation and YOLO training are offline workflows.

## Scope

Target organisms for v0:

- `Saccharomyces cerevisiae`
- `Pichia pastoris`, also known as `Komagataella phaffii`

Current common media and context:

- YPD yeast plates
- LSLB / `E. coli` selection contexts with antibiotic selection markers

Apricot is intended for preliminary yeast colony counting and still needs validation on real Dr. Ford plates before being treated as lab-grade.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Runtime dependencies are intentionally small for deployment:

- `opencv-python-headless`
- `pillow`
- `numpy`
- `gunicorn`

The base web app can boot and report health without loading YOLO. Install YOLO tooling only in an
inference/training environment that preserves `opencv-python-headless`; the base Render install avoids
packages whose default dependency set pulls in the GUI-enabled OpenCV wheel.

The project uses the existing stdlib HTTP server locally and exposes a WSGI `app:app` entrypoint for Gunicorn on Render.

## Run Locally

Place the default clean-dot specialist weights at:

```text
models/apricot_clean_dot_counter_v1.pt
```

Apricot can also use separately trained merged-colony specialist weights:

```text
models/apricot_merged_colony_counter_v1.pt
```

Set a custom path to select a specialist at runtime:

```bash
export APRICOT_MODEL_PATH=models/apricot_clean_dot_counter_v1.pt
export APRICOT_MERGED_MODEL_PATH=models/apricot_merged_colony_counter_v1.pt
```

Start the local server:

```bash
python scripts/serve_api.py --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Useful routes:

- `GET /`
- `GET /health`
- `POST /api/predict`
- `POST /api/predict-batch`
- `POST /predict`
- `POST /analyze`

If weights are missing, startup still succeeds and prediction requests return a clear service error. The health route reports configured model path, whether the file exists, whether the model is loaded, default confidence, upload limits, batch limits, schema support, and retention mode without training or loading YOLO.

Single-image API example:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "confidence=0.25" \
  -F "file=@plate1.jpg"
```

Batch API example:

```bash
curl -X POST http://127.0.0.1:8000/api/predict-batch \
  -F "confidence=0.25" \
  -F "images=@plate1.jpg" \
  -F "images=@plate2.png"
```

Supported upload types are JPG, JPEG, PNG, and WebP.

## Privacy And Retention

Default retention mode is:

```text
APRICOT_RETENTION_MODE=ephemeral
```

In ephemeral mode, uploaded images are processed transiently for counting and are not retained for training. Annotated images are returned to the browser as base64 data URLs and normal request artifacts are discarded. The service does not copy user uploads into `data/`, `data/generated/`, `eval_images/`, `runs/`, or any training dataset.

Optional local debugging:

```bash
export APRICOT_RETENTION_MODE=debug
```

Debug mode may keep request-scoped upload/output artifacts under `uploads/` and `outputs/` for inspection. Do not use debug retention for normal Render operation.

Batch and upload limits:

```text
APRICOT_MAX_UPLOAD_MB=12
APRICOT_MAX_BATCH_IMAGES=10
APRICOT_MAX_TOTAL_UPLOAD_MB=48
```

Current schema support:

- `clean_dots`: primary supported path.
- `merged_snowman`: accepted, with reduced reliability until more validation is available.
- `streak_lines`: coming later.

## Test And Smoke

```bash
make test
make generate-small
make generate-suite
make evaluate-synthetic
make run
make smoke-app
```

Equivalent direct commands:

```bash
python -m pytest
python scripts/smoke_generate.py --overwrite
python scripts/generate_synthetic.py --overwrite
python scripts/evaluate_synthetic_robustness.py
python scripts/smoke_app.py --base-url http://127.0.0.1:8000
```

Tests are fast and do not require real YOLO weights or training. Inference is monkeypatched where needed.

## Synthetic-Only Training Stage

We currently only have synthetic data. That is expected until Dr. Ford can provide live colony images.

Synthetic validation does **not** prove real-world accuracy. It only tells us whether the model learned this generator and whether it survives controlled synthetic perturbations. Treat these results as engineering signals for the training/evaluation pipeline, not as a lab-grade performance claim.

Apricot uses a three-tier synthetic protocol:

- `train_standard`: standard synthetic training images.
- `val_standard`: same-generator validation images. Useful for detecting training regressions, but still synthetic.
- Synthetic stress suite: out-of-distribution synthetic folders that perturb one regime at a time.

Stress splits:

- `test_lighting_shift`: brightness, contrast, gradients, and vignette changes.
- `test_agar_color_shift`: alternate dark agar color.
- `test_blur_compression`: defocus blur plus lower-quality JPEG compression.
- `test_density_extremes`: sparse and dense colony-count regimes.
- `test_size_extremes`: very small and very large colonies.
- `test_plate_position_shift`: petri dish shifted away from image center.
- `test_artifact_noise`: synthetic dust, scratches, and smudges.

Generate a small 12-plate smoke dataset:

```bash
python scripts/smoke_generate.py --out data/generated/apricot_smoke_12 --overwrite
```

Generate the full named train/validation/stress suite with one command:

```bash
python scripts/generate_synthetic.py \
  --out data/generated/apricot_synthetic_suite_v1 \
  --plates 100 \
  --stress-plates 20 \
  --img-size 2000 \
  --species s_cerevisiae \
  --medium YPD \
  --overwrite
```

Expected suite contract:

```text
data/generated/apricot_synthetic_suite_v1/
  dataset.yaml
  suite_manifest.json
  train_standard/
    dataset.yaml
    manifest.json
    images/
    labels/
  val_standard/
    dataset.yaml
    manifest.json
    images/
    labels/
  test_lighting_shift/
  test_agar_color_shift/
  test_blur_compression/
  test_density_extremes/
  test_size_extremes/
  test_plate_position_shift/
  test_artifact_noise/
```

The suite-level `dataset.yaml` trains on `train_standard/images` and validates on `val_standard/images`. Each split has its own `manifest.json` with generator parameters and `expected_colony_count` per image. Known counts come from generated labels/manifests, not manual input.

## Train Offline

Training is not part of the web service. Run it locally or in a training job:

```bash
python scripts/train_yolo.py \
  --schema clean_dots \
  --data data/generated/apricot_synthetic_suite_v1/dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --output models/apricot_clean_dot_counter_v1.pt
```

Train the merged-colony specialist from a dataset whose manifests declare `schema: merged_snowman`:

```bash
python scripts/train_yolo.py \
  --schema merged_snowman \
  --data data/generated/apricot_merged_snowman_suite_v1/dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --output models/apricot_merged_colony_counter_v1.pt
```

`scripts/train_yolo.py` defaults to clean-dot-only training and refuses merged-snowman or streak-line data on the clean-dot path. Use `--include-schema <schema>` only when intentionally mixing another schema into a separate experiment. The `merged_snowman` path defaults to `models/apricot_merged_colony_counter_v1.pt`, so it does not overwrite `models/apricot_clean_dot_counter_v1.pt`.

Ultralytics writes training runs under `runs/`. `scripts/train_yolo.py` copies the selected `best.pt` to the selected output path, or you can set `APRICOT_MODEL_PATH` to another trained weight file.

## Evaluate Synthetic Robustness

After training, evaluate all synthetic regimes with one command:

```bash
python scripts/evaluate_synthetic_robustness.py \
  --suite data/generated/apricot_synthetic_suite_v1 \
  --model models/apricot_clean_dot_counter_v1.pt \
  --out outputs/synthetic_robustness
```

The evaluator sweeps confidence thresholds:

```text
0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50
```

Outputs:

```text
outputs/synthetic_robustness/
  synthetic_robustness_per_image.csv
  synthetic_robustness_summary.csv
  annotated_samples/
```

The summary CSV reports mean absolute count error, mean percent count error, exact-count rate, and the best confidence threshold per split. Use it to see which synthetic regimes the model handles and where it fails.

## Transition To Real Dr. Ford Images

When live colony images arrive, keep this synthetic suite as a regression harness and add a separate real-image evaluation set:

- Store real images and annotations separately from generated data.
- Do not tune thresholds on the final real test set.
- Compare synthetic stress failures against real failure modes.
- Update training only after licensing, consent, and annotation format are clear.
- Report real-world accuracy only from real held-out Dr. Ford images, not from synthetic validation.

## Deploy On Render

`render.yaml` is configured for a Python web service:

```bash
pip install --upgrade pip && pip install .
gunicorn app:app --bind 0.0.0.0:$PORT
```

Render environment variable:

```text
APRICOT_MODEL_PATH=models/apricot_clean_dot_counter_v1.pt
APRICOT_MERGED_MODEL_PATH=models/apricot_merged_colony_counter_v1.pt
APRICOT_RETENTION_MODE=ephemeral
APRICOT_MAX_BATCH_IMAGES=10
```

Health check path:

```text
/health
```

The Render start command only launches the web server. It must not generate data, download datasets, or train YOLO.

## Repository Hygiene

Tracked placeholder directories:

- `uploads/.gitkeep`
- `outputs/.gitkeep`
- `models/.gitkeep`
- `data/generated/.gitkeep`
- `runs/.gitkeep`

Ignored runtime or generated content:

- uploads
- annotated outputs
- generated synthetic datasets
- model training datasets
- YOLO training runs
- model weight files

Do not commit large generated datasets, uploads, annotated outputs, training runs, or heavyweight model binaries. Keep reproducible commands and manifests in the repo instead.

## Legacy And Evaluation Utilities

The older explainable OpenCV pipeline remains available for reference and evaluation work. Annotation and benchmark helpers are still available:

- [CVAT label spec](CVAT_LABEL_SPEC.md)
- [Annotation schema](ANNOTATION_SCHEMA.md)
- [CVAT + Apricot integration plan](CVAT_GXP_PLAN.md)
- [Lab-grade roadmap](LAB_GRADE_ROADMAP.md)

For current deployment notes, see [APRICOT_RENDER_HANDOFF.md](APRICOT_RENDER_HANDOFF.md). For model limitations and intended use, see [docs/model_card.md](docs/model_card.md).
