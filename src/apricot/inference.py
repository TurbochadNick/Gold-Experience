from __future__ import annotations

import os
from io import BytesIO
from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .plate_schema import SCHEMA_CLEAN_DOTS, SCHEMA_MERGED_SNOWMAN

CLEAN_DOT_MODEL_PATH = Path("models/apricot_clean_dot_counter_v1.pt")
MERGED_SNOWMAN_MODEL_PATH = Path("models/apricot_merged_colony_counter_v1.pt")
DEFAULT_MODEL_PATH = CLEAN_DOT_MODEL_PATH
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.70
DEFAULT_MAX_DET = 1000
DEFAULT_LINE_WIDTH = 1

LOGGER = logging.getLogger(__name__)
_LOADED_MODEL_PATHS: set[str] = set()


class ModelNotFoundError(FileNotFoundError):
    """Raised when Apricot cannot find trained YOLO weights."""


class InferenceDependencyError(RuntimeError):
    """Raised when an optional inference dependency is unavailable."""


class ImageDecodeError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image."""


@dataclass(frozen=True)
class Detection:
    id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict[str, Any]:
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return {
            "id": self.id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "box": {
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2,
                "width": width,
                "height": height,
                "center_x": self.x1 + width / 2.0,
                "center_y": self.y1 + height / 2.0,
            },
        }


@dataclass
class PredictionResult:
    count: int
    detections: list[Detection]
    confidence_threshold: float
    iou_threshold: float
    max_det: int
    model_path: Path
    model_version: str
    image_width: int
    image_height: int
    annotated_image: np.ndarray
    annotated_image_path: Path | None = None
    annotated_image_bytes: bytes | None = None

    def to_dict(self, *, include_image_bytes: bool = False) -> dict[str, Any]:
        detections = [detection.to_dict() for detection in self.detections]
        payload: dict[str, Any] = {
            "count": self.count,
            "detections": detections,
            "boxes": [
                {
                    "id": detection["id"],
                    "x1": detection["x1"],
                    "y1": detection["y1"],
                    "x2": detection["x2"],
                    "y2": detection["y2"],
                    "confidence": detection["confidence"],
                    "class_id": detection["class_id"],
                    "class_name": detection["class_name"],
                }
                for detection in detections
            ],
            "confidence_scores": [detection["confidence"] for detection in detections],
            "threshold_used": self.confidence_threshold,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "max_det": self.max_det,
            "model_path": public_model_path(self.model_path),
            "model_version": self.model_version,
            "model": {
                "path": public_model_path(self.model_path),
                "version": self.model_version,
            },
            "image": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "annotated_image_path": public_model_path(self.annotated_image_path) if self.annotated_image_path else None,
        }
        if include_image_bytes and self.annotated_image_bytes is not None:
            payload["annotated_image_bytes"] = self.annotated_image_bytes
        return payload


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    selected = model_path or os.environ.get("APRICOT_MODEL_PATH") or DEFAULT_MODEL_PATH
    return Path(selected).expanduser()


def model_schema_for_route(schema: str | None) -> str:
    if schema == SCHEMA_MERGED_SNOWMAN:
        return SCHEMA_MERGED_SNOWMAN
    return SCHEMA_CLEAN_DOTS


def resolve_model_path_for_schema(schema: str | None) -> Path:
    if model_schema_for_route(schema) == SCHEMA_MERGED_SNOWMAN:
        selected = os.environ.get("APRICOT_MERGED_MODEL_PATH") or MERGED_SNOWMAN_MODEL_PATH
        return resolve_model_path(selected)
    return resolve_model_path()


def public_model_path(model_path: str | Path | None = None) -> str:
    path = resolve_model_path(model_path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return path.name


def model_version(model_path: str | Path | None = None) -> str:
    path = resolve_model_path(model_path)
    if not path.is_file():
        return "missing"
    try:
        stat = path.stat()
    except OSError:
        return path.name
    return f"{path.name}:{stat.st_mtime_ns}"


def model_status(model_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_model_path(model_path)
    cache_key = str(path)
    return {
        "model_path": public_model_path(path),
        "model_version": model_version(path),
        "model_exists": path.is_file(),
        "model_loaded": cache_key in _LOADED_MODEL_PATHS,
    }


def model_missing_message(model_path: str | Path | None = None) -> str:
    path = resolve_model_path(model_path)
    if path.name == MERGED_SNOWMAN_MODEL_PATH.name:
        return (
            f"Model weights not found at {public_model_path(path)}. "
            "Set APRICOT_MERGED_MODEL_PATH or place weights at "
            "models/apricot_merged_colony_counter_v1.pt."
        )
    return (
        f"Model weights not found at {public_model_path(path)}. "
        "Set APRICOT_MODEL_PATH or place weights at models/apricot_clean_dot_counter_v1.pt."
    )


@lru_cache(maxsize=4)
def load_model(model_path: str) -> Any:
    path = resolve_model_path(model_path)
    if not path.is_file():
        raise ModelNotFoundError(model_missing_message(path))

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on deployment image
        raise InferenceDependencyError(
            "Ultralytics is not installed. Install runtime dependencies before running inference."
        ) from exc

    LOGGER.info("Loading Apricot YOLO model path=%s version=%s", public_model_path(path), model_version(path))
    model = YOLO(str(path))
    _LOADED_MODEL_PATHS.add(str(path))
    return model


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    try:
        with Image.open(BytesIO(image_bytes)) as pil_image:
            rgb_image = pil_image.convert("RGB")
            rgb_array = np.array(rgb_image)
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageDecodeError("Uploaded file is not a readable image.") from exc

    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_numpy(value: Any) -> np.ndarray:
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    return np.asarray(value)


def _detections_from_result(result: Any) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = getattr(result, "names", {}) or {}
    xyxy = _to_numpy(getattr(boxes, "xyxy", [])).reshape((-1, 4))
    if xyxy.size == 0:
        return []
    confs = _to_numpy(getattr(boxes, "conf", np.zeros(len(xyxy), dtype=float))).reshape((-1,))
    class_ids = _to_numpy(getattr(boxes, "cls", np.zeros(len(xyxy), dtype=int))).astype(int).reshape((-1,))

    detections: list[Detection] = []
    for index, (box, confidence, class_id) in enumerate(zip(xyxy, confs, class_ids, strict=False), start=1):
        x1, y1, x2, y2 = [float(value) for value in box]
        detections.append(
            Detection(
                id=index,
                class_id=int(class_id),
                class_name=str(names.get(int(class_id), "yeast_colony")),
                confidence=float(confidence),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return detections


def render_clean_annotation(
    image: np.ndarray,
    detections: list[Detection],
    *,
    line_width: int = DEFAULT_LINE_WIDTH,
) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    stroke = max(1, min(2, int(line_width)))
    color = (45, 132, 211)
    for detection in detections:
        x1 = int(round(_clamp(detection.x1, 0, width - 1)))
        y1 = int(round(_clamp(detection.y1, 0, height - 1)))
        x2 = int(round(_clamp(detection.x2, 0, width - 1)))
        y2 = int(round(_clamp(detection.y2, 0, height - 1)))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, stroke)
    return annotated


def predict_image_array(
    image: np.ndarray,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    max_det: int = DEFAULT_MAX_DET,
    model_path: str | Path | None = None,
    output_dir: Path | None = None,
    output_stem: str = "apricot_prediction",
) -> PredictionResult:
    resolved_model_path = resolve_model_path(model_path)
    confidence = _clamp(float(confidence), 0.01, 0.99)
    iou = _clamp(float(iou), 0.01, 0.99)
    max_det = max(1, int(max_det))

    model = load_model(str(resolved_model_path))
    results = model.predict(
        source=image,
        conf=confidence,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )
    result = results[0]
    detections = _detections_from_result(result)
    annotated = render_clean_annotation(image, detections)

    annotated_path: Path | None = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = output_dir / f"{output_stem}.jpg"
        cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 94])

    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 94])
    annotated_bytes = encoded.tobytes() if ok else None
    height, width = image.shape[:2]

    return PredictionResult(
        count=len(detections),
        detections=detections,
        confidence_threshold=confidence,
        iou_threshold=iou,
        max_det=max_det,
        model_path=resolved_model_path,
        model_version=model_version(resolved_model_path),
        image_width=width,
        image_height=height,
        annotated_image=annotated,
        annotated_image_path=annotated_path,
        annotated_image_bytes=annotated_bytes,
    )


def predict_colonies(
    image_path: str | Path,
    conf: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    max_det: int = DEFAULT_MAX_DET,
    *,
    model_path: str | Path | None = None,
    output_dir: Path | None = None,
    output_stem: str | None = None,
) -> dict[str, Any]:
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        image = decode_image_bytes(path.read_bytes())
    result = predict_image_array(
        image,
        confidence=conf,
        iou=iou,
        max_det=max_det,
        model_path=model_path,
        output_dir=output_dir,
        output_stem=output_stem or f"{path.stem}-annotated",
    )
    return result.to_dict(include_image_bytes=output_dir is None)


def predict_image_bytes(
    image_bytes: bytes,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    max_det: int = DEFAULT_MAX_DET,
    model_path: str | Path | None = None,
    output_dir: Path | None = None,
    output_stem: str = "apricot_prediction",
) -> PredictionResult:
    image = decode_image_bytes(image_bytes)
    return predict_image_array(
        image,
        confidence=confidence,
        iou=iou,
        max_det=max_det,
        model_path=model_path,
        output_dir=output_dir,
        output_stem=output_stem,
    )
