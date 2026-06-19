from __future__ import annotations

import base64
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from apricot.inference import (
    CLEAN_DOT_MODEL_PATH,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_MAX_DET,
    decode_image_bytes,
    ImageDecodeError,
    InferenceDependencyError,
    MERGED_SNOWMAN_MODEL_PATH,
    ModelNotFoundError,
    model_schema_for_route,
    model_missing_message,
    model_status,
    model_version,
    predict_image_array,
    public_model_path,
    resolve_model_path_for_schema,
)
from apricot.schema_router import route_image_schema
from apricot import __version__ as APRICOT_VERSION

ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web"
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_SCHEMA_LABELS = ("clean_dots", "merged_snowman")
COMING_SOON_SCHEMA_LABELS = ("streak_lines",)
MODEL_SPECIALIST_LABELS = {
    "clean_dots": "Clean-dot counter",
    "merged_snowman": "Merged-colony counter",
}
IMAGE_CONTENT_TYPE_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
MAX_UPLOAD_BYTES = int(
    os.environ.get(
        "APRICOT_MAX_UPLOAD_BYTES",
        str(int(float(os.environ.get("APRICOT_MAX_UPLOAD_MB", "12")) * 1024 * 1024)),
    )
)
MAX_BATCH_IMAGES = int(os.environ.get("APRICOT_MAX_BATCH_IMAGES", "10"))
MAX_TOTAL_UPLOAD_BYTES = int(
    os.environ.get(
        "APRICOT_MAX_TOTAL_UPLOAD_BYTES",
        str(int(float(os.environ.get("APRICOT_MAX_TOTAL_UPLOAD_MB", "48")) * 1024 * 1024)),
    )
)

CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class UploadFile:
    field_name: str
    filename: str | None
    content_type: str | None
    payload: bytes


def ensure_runtime_dirs() -> None:
    for directory in (UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def retention_mode() -> str:
    configured = os.environ.get("APRICOT_RETENTION_MODE", "ephemeral").strip().lower() or "ephemeral"
    return configured if configured in {"ephemeral", "debug"} else "ephemeral"


def should_keep_runtime_outputs() -> bool:
    return retention_mode() == "debug"


def health_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "status": "ok",
        "product": "Apricot Colony Counter",
        "engine": "apricot-colony-counter",
        "version": APRICOT_VERSION,
        "default_confidence": DEFAULT_CONFIDENCE,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_batch_images": MAX_BATCH_IMAGES,
        "max_total_upload_bytes": MAX_TOTAL_UPLOAD_BYTES,
        "retention_mode": retention_mode(),
        "allowed_image_types": sorted(ALLOWED_IMAGE_SUFFIXES),
        "supported_schemas": list(SUPPORTED_SCHEMA_LABELS),
        "coming_soon_schemas": list(COMING_SOON_SCHEMA_LABELS),
        "model_specialists": model_specialists_payload(),
        **model_status(),
    }


def model_specialists_payload() -> dict[str, dict[str, Any]]:
    defaults = {
        "clean_dots": CLEAN_DOT_MODEL_PATH,
        "merged_snowman": MERGED_SNOWMAN_MODEL_PATH,
    }
    env_vars = {
        "clean_dots": "APRICOT_MODEL_PATH",
        "merged_snowman": "APRICOT_MERGED_MODEL_PATH",
    }
    specialists: dict[str, dict[str, Any]] = {}
    for schema in SUPPORTED_SCHEMA_LABELS:
        path = resolve_model_path_for_schema(schema)
        status = model_status(path)
        specialists[schema] = {
            "schema": schema,
            "display_name": MODEL_SPECIALIST_LABELS[schema],
            "path": status["model_path"],
            "default_path": public_model_path(defaults[schema]),
            "env_var": env_vars[schema],
            "model_exists": status["model_exists"],
            "model_loaded": status["model_loaded"],
            "model_version": status["model_version"],
        }
    return specialists


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")


def _safe_suffix(filename: str | None, content_type: str | None = None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in ALLOWED_IMAGE_SUFFIXES:
        return suffix
    normalized_content_type = (content_type or "").split(";", 1)[0].strip().lower()
    if not suffix and normalized_content_type in IMAGE_CONTENT_TYPE_SUFFIXES:
        return IMAGE_CONTENT_TYPE_SUFFIXES[normalized_content_type]
    raise ValueError("Unsupported file type. Upload a JPG, PNG, or WebP image.")


def _is_allowed_upload(filename: str | None, content_type: str | None) -> bool:
    try:
        _safe_suffix(filename, content_type)
    except ValueError:
        return False
    return True


def _public_filename(filename: str | None, fallback: str = "upload") -> str:
    cleaned = Path(filename or fallback).name
    return cleaned or fallback


def _mime_type_for_suffix(suffix: str) -> str:
    return CONTENT_TYPES.get(suffix.lower(), "image/jpeg").split(";", 1)[0]


def _annotated_image_payload(image_bytes: bytes | None, mime_type: str = "image/jpeg") -> dict[str, str]:
    if not image_bytes:
        return {}
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {
        "annotated_image_base64": encoded,
        "annotated_image_data_url": f"data:{mime_type};base64,{encoded}",
        "mime_type": mime_type,
    }


def _schema_reliability_warning(schema: str) -> str | None:
    if schema in SUPPORTED_SCHEMA_LABELS:
        return None
    if schema == "streak_lines":
        return (
            "Reliability is reduced: streak-line plates are not supported by the current counting model yet."
        )
    return (
        f"Reliability is reduced for {schema} plates. Current validation is strongest for clean_dots."
    )


def _parse_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    parsed = float(value)
    if parsed > 1.0:
        parsed = parsed / 100.0
    return parsed


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_multipart_payload_many(
    body: bytes,
    content_type: str,
) -> tuple[list[UploadFile], dict[str, str]]:
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\n\r\n" + body
    )
    fields: dict[str, str] = {}
    if not message.is_multipart():
        return [], fields

    files: list[UploadFile] = []
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        field_name = part.get_param("name", header="Content-Disposition")
        if not field_name:
            continue

        part_filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if part_filename is not None and payload:
            files.append(
                UploadFile(
                    field_name=str(field_name),
                    filename=part_filename,
                    content_type=part.get_content_type(),
                    payload=payload,
                )
            )
            continue

        charset = part.get_content_charset() or "utf-8"
        fields[field_name] = payload.decode(charset, errors="replace").strip()

    return files, fields


def _parse_multipart_payload(
    body: bytes,
    content_type: str,
) -> tuple[bytes | None, str | None, str | None, dict[str, str]]:
    files, fields = _parse_multipart_payload_many(body, content_type)
    if not files:
        return None, None, None, fields
    first = files[0]
    return first.payload, first.filename, first.content_type, fields


def _prediction_payload(
    *,
    file_bytes: bytes,
    filename: str | None,
    content_type: str | None,
    confidence: float,
    iou: float,
    max_det: int,
    include_images: bool = True,
) -> dict[str, Any]:
    suffix = _safe_suffix(filename, content_type)
    image = decode_image_bytes(file_bytes)
    route = route_image_schema(image)
    selected_schema = model_schema_for_route(route.schema)
    selected_model_path = resolve_model_path_for_schema(route.schema)
    selected_model = {
        "schema": selected_schema,
        "path": public_model_path(selected_model_path),
        "version": model_version(selected_model_path),
    }

    upload_id = uuid4().hex
    output_dir: Path | None = None
    if should_keep_runtime_outputs():
        upload_dir = UPLOAD_DIR / upload_id
        output_dir = OUTPUT_DIR / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / f"original{suffix}").write_bytes(file_bytes)

    started_at = perf_counter()
    result = predict_image_array(
        image,
        confidence=confidence,
        iou=iou,
        max_det=max_det,
        model_path=selected_model_path,
        output_dir=output_dir,
        output_stem="annotated",
    )
    duration_ms = (perf_counter() - started_at) * 1000.0
    payload = result.to_dict()
    annotated_url = None
    annotated_image_path = getattr(result, "annotated_image_path", None)
    if should_keep_runtime_outputs() and annotated_image_path:
        annotated_url = f"/outputs/{upload_id}/{annotated_image_path.name}"
    annotated_image_bytes = getattr(result, "annotated_image_bytes", None)
    image_payload = _annotated_image_payload(annotated_image_bytes) if include_images else {}
    warning = _schema_reliability_warning(route.schema)
    payload.update(
        {
            "ok": True,
            "filename": _public_filename(filename),
            "upload_id": upload_id,
            "duration_ms": round(duration_ms, 1),
            "annotated_image_url": annotated_url,
            **image_payload,
            "schema": route.schema,
            "route_schema": route.schema,
            "selected_schema": selected_schema,
            "chosen_schema": selected_schema,
            "selected_model": selected_model,
            "chosen_model": selected_model,
            "confidence": route.confidence,
            "schema_confidence": route.confidence,
            "route": route.to_dict(),
            "route_metadata": route.metadata,
            "supported_schemas": list(SUPPORTED_SCHEMA_LABELS),
            "coming_soon_schemas": list(COMING_SOON_SCHEMA_LABELS),
            "retention_mode": retention_mode(),
            "reliability_warning": warning,
        }
    )
    if isinstance(payload.get("model"), dict):
        payload["model"].update({"schema": selected_schema})
    else:
        payload["model"] = dict(selected_model)
    print(
        "Apricot prediction "
        f"upload_id={upload_id} count={result.count} "
        f"schema={route.schema} selected_model={selected_model['path']} route_confidence={route.confidence:.2f} "
        f"threshold={result.confidence_threshold:.2f} duration_ms={duration_ms:.1f}"
    )
    return payload


def _batch_prediction_payload(
    *,
    files: list[UploadFile],
    fields: dict[str, str],
) -> tuple[HTTPStatus, dict[str, Any]]:
    if not files:
        return HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Expected one or more image files."}
    if len(files) > MAX_BATCH_IMAGES:
        return (
            HTTPStatus.BAD_REQUEST,
            {
                "ok": False,
                "error": f"Too many images. Maximum batch size is {MAX_BATCH_IMAGES}.",
                "max_batch_images": MAX_BATCH_IMAGES,
            },
        )

    confidence = _parse_float(
        fields.get("confidence") or fields.get("conf") or fields.get("threshold"),
        DEFAULT_CONFIDENCE,
    )
    iou = _parse_float(fields.get("iou"), DEFAULT_IOU)
    max_det = _parse_int(fields.get("max_det"), DEFAULT_MAX_DET)
    include_images = _parse_bool(fields.get("include_images"), True)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    service_error: Exception | None = None

    for index, upload in enumerate(files, start=1):
        filename = _public_filename(upload.filename, fallback=f"image-{index}")
        try:
            suffix = _safe_suffix(upload.filename, upload.content_type)
            result = _prediction_payload(
                file_bytes=upload.payload,
                filename=upload.filename,
                content_type=upload.content_type,
                confidence=confidence,
                iou=iou,
                max_det=max_det,
                include_images=include_images,
            )
            result.update(
                {
                    "ok": True,
                    "filename": filename,
                    "mime_type": _mime_type_for_suffix(suffix),
                }
            )
            results.append(result)
        except (ValueError, ImageDecodeError) as exc:
            errors.append({"filename": filename, "ok": False, "error": str(exc)})
        except (ModelNotFoundError, InferenceDependencyError) as exc:
            service_error = exc
            errors.append({"filename": filename, "ok": False, "error": str(exc) or model_missing_message()})

    count_total = sum(int(result.get("count", 0)) for result in results)
    payload: dict[str, Any] = {
        "ok": bool(results),
        "count_total": count_total,
        "images_processed": len(results),
        "images_failed": len(errors),
        "confidence": confidence,
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "max_det": max_det,
        "retention_mode": retention_mode(),
        "supported_schemas": list(SUPPORTED_SCHEMA_LABELS),
        "coming_soon_schemas": list(COMING_SOON_SCHEMA_LABELS),
        "results": results,
        "errors": errors,
    }
    if results:
        status = HTTPStatus.OK
    elif service_error is not None:
        status = HTTPStatus.SERVICE_UNAVAILABLE
        payload["error"] = str(service_error) or model_missing_message()
    else:
        status = HTTPStatus.BAD_REQUEST
        payload["error"] = "No images could be processed."
    return status, payload


def _render_result_html(payload: dict[str, Any]) -> bytes:
    if not payload.get("ok"):
        body = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Apricot Colony Counter</title>
    <link rel="stylesheet" href="/styles.css" />
  </head>
  <body>
    <main class="server-result">
      <h1>Apricot Colony Counter</h1>
      <p class="error-text">{escape(str(payload.get("error", "Prediction failed.")))}</p>
      <a class="button-link" href="/">Back to upload</a>
    </main>
  </body>
</html>
"""
        return body.encode("utf-8")

    annotated_url = escape(str(payload.get("annotated_image_url") or ""))
    count = escape(str(payload.get("count", 0)))
    confidence = float(payload.get("confidence_threshold", 0.0))
    selected_schema = escape(str(payload.get("selected_schema") or payload.get("schema") or "clean_dots"))
    selected_model = payload.get("selected_model") if isinstance(payload.get("selected_model"), dict) else {}
    selected_model_path = escape(str(selected_model.get("path") or payload.get("model_path") or ""))
    body = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Apricot Colony Counter</title>
    <link rel="stylesheet" href="/styles.css" />
  </head>
  <body>
    <main class="server-result">
      <h1>Apricot Colony Counter</h1>
      <p class="server-count">{count}</p>
      <p>Counted as: {selected_schema}</p>
      <p>Model: {selected_model_path}</p>
      <p>Confidence threshold: {confidence:.2f}</p>
      <img src="{annotated_url}" alt="Annotated colony detection result" />
      <a class="button-link" download href="{annotated_url}">Download annotated image</a>
      <a class="button-link secondary-link" href="/">Run another plate</a>
    </main>
  </body>
</html>
"""
    return body.encode("utf-8")


class ApricotRequestHandler(BaseHTTPRequestHandler):
    server_version = "ApricotHTTP/0.2"

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _parse_upload(self) -> tuple[bytes | None, str | None, str | None, dict[str, str]]:
        content_type = self.headers.get("Content-Type", "")
        try:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if content_length <= 0:
            return None, None, None, {}
        if content_length > MAX_UPLOAD_BYTES:
            limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
            raise ValueError(f"Upload is too large. Maximum size is {limit_mb:.0f} MB.")

        payload = self.rfile.read(content_length)
        if "multipart/form-data" in content_type:
            return _parse_multipart_payload(payload, content_type)
        if content_type.startswith("image/"):
            return payload, self.headers.get("X-Filename"), content_type, {}
        return None, None, None, {}

    def _parse_uploads(self) -> tuple[list[UploadFile], dict[str, str]]:
        content_type = self.headers.get("Content-Type", "")
        try:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if content_length <= 0:
            return [], {}
        if content_length > MAX_TOTAL_UPLOAD_BYTES:
            limit_mb = MAX_TOTAL_UPLOAD_BYTES / (1024 * 1024)
            raise ValueError(f"Upload batch is too large. Maximum total size is {limit_mb:.0f} MB.")

        payload = self.rfile.read(content_length)
        if "multipart/form-data" in content_type:
            return _parse_multipart_payload_many(payload, content_type)
        if content_type.startswith("image/"):
            filename = self.headers.get("X-Filename")
            return [UploadFile("file", filename, content_type, payload)], {}
        return [], {}

    def _serve_file_from_root(self, root: Path, request_path: str, prefix: str = "") -> bool:
        relative_path = request_path.removeprefix(prefix).lstrip("/") or "index.html"
        if relative_path.endswith("/"):
            relative_path = f"{relative_path}index.html"

        root_resolved = root.resolve()
        candidate = (root / relative_path).resolve()
        if root_resolved not in candidate.parents and candidate != root_resolved:
            return False
        if not candidate.is_file():
            return False

        content_type = CONTENT_TYPES.get(candidate.suffix.lower(), "application/octet-stream")
        self._send_bytes(HTTPStatus.OK, candidate.read_bytes(), content_type)
        return True

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(HTTPStatus.OK, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path.rstrip("/") == "/health":
            self._send_json(HTTPStatus.OK, health_payload())
            return

        if path.startswith("/outputs/"):
            if self._serve_file_from_root(OUTPUT_DIR, path, prefix="/outputs"):
                return

        if path == "/" or path in {"/index.html", "/app.js", "/styles.css", "/apricot-icon.svg"}:
            if self._serve_file_from_root(WEB_ROOT, path):
                return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path).rstrip("/")
        wants_html = path == "/predict"
        wants_json = path in {"/api/predict", "/analyze"}
        wants_batch = path == "/api/predict-batch"
        if not wants_html and not wants_json and not wants_batch:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return

        try:
            if wants_batch:
                files, fields = self._parse_uploads()
                status, payload = _batch_prediction_payload(files=files, fields=fields)
            else:
                file_bytes, filename, content_type, fields = self._parse_upload()
                if not file_bytes:
                    payload = {"ok": False, "error": "Expected multipart form-data with a file field."}
                    status = HTTPStatus.BAD_REQUEST
                elif not _is_allowed_upload(filename, content_type):
                    payload = {
                        "ok": False,
                        "error": "Unsupported upload type. Use a JPEG, PNG, or WebP image.",
                    }
                    status = HTTPStatus.UNSUPPORTED_MEDIA_TYPE
                else:
                    confidence = _parse_float(
                        fields.get("confidence") or fields.get("conf") or fields.get("threshold"),
                        DEFAULT_CONFIDENCE,
                    )
                    iou = _parse_float(fields.get("iou"), DEFAULT_IOU)
                    max_det = _parse_int(fields.get("max_det"), DEFAULT_MAX_DET)
                    include_images = _parse_bool(fields.get("include_images"), True)
                    payload = _prediction_payload(
                        file_bytes=file_bytes,
                        filename=filename,
                        content_type=content_type,
                        confidence=confidence,
                        iou=iou,
                        max_det=max_det,
                        include_images=include_images,
                    )
                    status = HTTPStatus.OK
        except (ValueError, ImageDecodeError) as exc:
            payload = {"ok": False, "error": str(exc)}
            status = HTTPStatus.BAD_REQUEST
        except (ModelNotFoundError, InferenceDependencyError) as exc:
            payload = {"ok": False, "error": str(exc) or model_missing_message()}
            status = HTTPStatus.SERVICE_UNAVAILABLE
        except Exception as exc:  # pragma: no cover - defensive API surface
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            status = HTTPStatus.INTERNAL_SERVER_ERROR

        if wants_html:
            self._send_bytes(status, _render_result_html(payload), "text/html; charset=utf-8")
            return
        self._send_json(status, payload)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    ensure_runtime_dirs()
    server = ThreadingHTTPServer((host, port), ApricotRequestHandler)
    try:
        print(f"Apricot Colony Counter listening on http://{host}:{port}")
        status = model_status()
        if status["model_exists"]:
            print(f"Using Apricot model {status['model_path']} version={status['model_version']}")
        else:
            print(model_missing_message())
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
