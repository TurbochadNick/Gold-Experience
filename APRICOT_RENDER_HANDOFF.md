# Apricot Render Handoff

This answers the integration questions for the current Render-facing web service.

## Current Service

- Product name: Apricot Colony Counter
- Python package distribution: `apricot-colony-counter`
- WSGI application entrypoint: `app:app`
- HTTP/WSGI implementation: Apricot's lightweight WSGI service exposed through `app:app`
- Web framework: no Flask/FastAPI/Django. The deployed app exposes a lightweight WSGI callable for Gunicorn.
- Frontend files: `web/index.html`, `web/app.js`, `web/styles.css`
- Container config: `Dockerfile`
- Render start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
- Host/port: Gunicorn binds to Render's `$PORT`.

## Dependencies

Declared in `pyproject.toml`:

```toml
dependencies = [
  "numpy>=2.0",
  "opencv-python-headless>=4.10",
  "pillow>=10.0",
  "gunicorn>=22.0",
]
```

There is no `requirements.txt` for the deployed service. Render uses `render.yaml`.
The base Render install intentionally excludes packages whose default dependency metadata installs the
GUI-enabled OpenCV wheel. Keep Render boot/health checks on headless OpenCV only; install YOLO tooling
in a separate inference/training image or environment that does not replace headless OpenCV.

## Current Inference Path

The current deployed inference is YOLO-backed through `src/apricot/inference.py`.

- Default clean-dot weights: `models/apricot_clean_dot_counter_v1.pt`
- Optional merged-colony weights: `models/apricot_merged_colony_counter_v1.pt`
- Route/schema selection: `src/apricot/schema_router.py`
- WSGI app: `app:app`

## API Endpoints

### `GET /health`

Returns:

```json
{
  "ok": true,
  "product": "Apricot Colony Counter",
  "engine": "apricot-colony-counter"
}
```

### `POST /api/predict`

Accepts either:

- `multipart/form-data` with field name `file`, `image`, or `upload`
- raw image bytes with an `image/*` content type

Processing:

1. Decode upload with OpenCV.
2. Route image schema.
3. Select the clean-dot or merged-colony model.
4. Run YOLO inference.
5. Return count, boxes, schema metadata, and an annotated preview.

Returns JSON:

```json
{
  "ok": true,
  "count": 42,
  "selected_schema": "clean_dots",
  "selected_model": {"path": "models/apricot_clean_dot_counter_v1.pt"},
  "detections": []
}
```

The browser renders returned annotated image previews and detection rows.

## Model Status

The deployed service defaults to the clean-dot model path:

```text
APRICOT_MODEL_PATH=models/apricot_clean_dot_counter_v1.pt
```

Prediction requests return a clear service error if weights are missing. `/health` reports model existence without loading the model.

## Synthetic Data Workflow

The Perplexity notebook has been adapted into a repo script:

```bash
python3 scripts/train_apricot_yolo_synthetic.py
```

Default outputs are ignored by git:

```text
data/generated/apricot_synthetic_v3/
data/model_training/apricot_yolo_synthetic_v3/
```

Train YOLO locally, not on Render:

```bash
# Install YOLO tooling in a local training environment that preserves opencv-python-headless.
python3 scripts/train_apricot_yolo_synthetic.py --train
```

Expected trained weights path:

```text
runs/detect/apricot_v3_realistic/weights/best.pt
```

Do not use `AGAR_representative` as training data for commercial work. The synthetic generator is self-contained and does not depend on AGAR images or annotations.
