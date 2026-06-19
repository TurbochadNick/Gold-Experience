FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    APRICOT_MODEL_PATH=models/apricot_clean_dot_counter_v1.pt

WORKDIR /app

COPY pyproject.toml README.md app.py ./
COPY src ./src
COPY scripts ./scripts
COPY web ./web
COPY models ./models

RUN python -m pip install --upgrade pip \
    && python -m pip install . \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=1.8.0" \
        "torchvision>=0.9.0" \
    && python -m pip install \
        "matplotlib>=3.3.0" \
        "pyyaml>=5.3.1" \
        "requests>=2.23.0" \
        "scipy>=1.4.1" \
        "psutil>=5.8.0" \
        "polars>=0.20.0" \
        "nvidia-ml-py>=12.0.0" \
        "ultralytics-thop>=2.0.18" \
    && python -m pip install "ultralytics>=8.3" --no-deps \
    && python -m pip show opencv-python-headless \
    && ! python -m pip show opencv-python

EXPOSE 8000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
