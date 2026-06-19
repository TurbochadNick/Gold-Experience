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

RUN pip install --upgrade pip && pip install .

EXPOSE 8000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
