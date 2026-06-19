from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import unquote

from gold_experience import api_server

StartResponse = Callable[[str, list[tuple[str, str]]], None]


def _status_line(status: HTTPStatus) -> str:
    return f"{status.value} {status.phrase}"


def _send(
    start_response: StartResponse,
    status: HTTPStatus,
    body: bytes,
    content_type: str,
    *,
    cors: bool = False,
) -> Iterable[bytes]:
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),
    ]
    if cors:
        headers.extend(
            [
                ("Access-Control-Allow-Origin", "*"),
                ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
                ("Access-Control-Allow-Headers", "Content-Type"),
            ]
        )
    start_response(_status_line(status), headers)
    return [body]


def _json_response(start_response: StartResponse, status: HTTPStatus, payload: dict[str, Any]) -> Iterable[bytes]:
    return _send(
        start_response,
        status,
        api_server._json_bytes(payload),
        "application/json; charset=utf-8",
        cors=True,
    )


def _serve_file(start_response: StartResponse, root: Path, request_path: str, prefix: str = "") -> Iterable[bytes] | None:
    relative_path = request_path.removeprefix(prefix).lstrip("/") or "index.html"
    if relative_path.endswith("/"):
        relative_path = f"{relative_path}index.html"

    root_resolved = root.resolve()
    candidate = (root / relative_path).resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        return None
    if not candidate.is_file():
        return None

    content_type = api_server.CONTENT_TYPES.get(candidate.suffix.lower(), "application/octet-stream")
    return _send(start_response, HTTPStatus.OK, candidate.read_bytes(), content_type)


def _parse_wsgi_upload(environ: dict[str, Any]) -> tuple[bytes | None, str | None, str | None, dict[str, str]]:
    content_type = str(environ.get("CONTENT_TYPE", ""))
    content_length = int(environ.get("CONTENT_LENGTH") or "0")
    if content_length <= 0:
        return None, None, None, {}
    if content_length > api_server.MAX_UPLOAD_BYTES:
        limit_mb = api_server.MAX_UPLOAD_BYTES / (1024 * 1024)
        raise ValueError(f"Upload is too large. Maximum size is {limit_mb:.0f} MB.")

    body = environ["wsgi.input"].read(content_length)
    if "multipart/form-data" in content_type:
        return api_server._parse_multipart_payload(body, content_type)
    if content_type.startswith("image/"):
        filename = str(environ.get("HTTP_X_FILENAME", "")) or None
        return body, filename, content_type, {}
    return None, None, None, {}


def _parse_wsgi_uploads(environ: dict[str, Any]) -> tuple[list[api_server.UploadFile], dict[str, str]]:
    content_type = str(environ.get("CONTENT_TYPE", ""))
    content_length = int(environ.get("CONTENT_LENGTH") or "0")
    if content_length <= 0:
        return [], {}
    if content_length > api_server.MAX_TOTAL_UPLOAD_BYTES:
        limit_mb = api_server.MAX_TOTAL_UPLOAD_BYTES / (1024 * 1024)
        raise ValueError(f"Upload batch is too large. Maximum total size is {limit_mb:.0f} MB.")

    body = environ["wsgi.input"].read(content_length)
    if "multipart/form-data" in content_type:
        return api_server._parse_multipart_payload_many(body, content_type)
    if content_type.startswith("image/"):
        filename = str(environ.get("HTTP_X_FILENAME", "")) or None
        return [api_server.UploadFile("file", filename, content_type, body)], {}
    return [], {}


def _handle_prediction(
    environ: dict[str, Any],
    start_response: StartResponse,
    *,
    wants_html: bool,
) -> Iterable[bytes]:
    content_type = str(environ.get("CONTENT_TYPE", ""))
    try:
        file_bytes, filename, file_content_type, fields = _parse_wsgi_upload(environ)
        if not file_bytes:
            payload = {"ok": False, "error": "Expected multipart form-data with a file field."}
            status = HTTPStatus.BAD_REQUEST
        elif not api_server._is_allowed_upload(filename, file_content_type):
            payload = {
                "ok": False,
                "error": "Unsupported upload type. Use a JPEG, PNG, or WebP image.",
            }
            status = HTTPStatus.UNSUPPORTED_MEDIA_TYPE
        else:
            confidence = api_server._parse_float(
                fields.get("confidence") or fields.get("conf") or fields.get("threshold"),
                api_server.DEFAULT_CONFIDENCE,
            )
            iou = api_server._parse_float(fields.get("iou"), api_server.DEFAULT_IOU)
            max_det = api_server._parse_int(fields.get("max_det"), api_server.DEFAULT_MAX_DET)
            include_images = api_server._parse_bool(fields.get("include_images"), True)
            payload = api_server._prediction_payload(
                file_bytes=file_bytes,
                filename=filename,
                content_type=file_content_type,
                confidence=confidence,
                iou=iou,
                max_det=max_det,
                include_images=include_images,
            )
            status = HTTPStatus.OK
    except (ValueError, api_server.ImageDecodeError) as exc:
        payload = {"ok": False, "error": str(exc)}
        status = HTTPStatus.BAD_REQUEST
    except (api_server.ModelNotFoundError, api_server.InferenceDependencyError) as exc:
        payload = {"ok": False, "error": str(exc) or api_server.model_missing_message()}
        status = HTTPStatus.SERVICE_UNAVAILABLE
    except Exception as exc:  # pragma: no cover - defensive WSGI surface
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        status = HTTPStatus.INTERNAL_SERVER_ERROR

    if wants_html:
        return _send(start_response, status, api_server._render_result_html(payload), "text/html; charset=utf-8")
    return _json_response(start_response, status, payload)


def _handle_batch_prediction(environ: dict[str, Any], start_response: StartResponse) -> Iterable[bytes]:
    try:
        files, fields = _parse_wsgi_uploads(environ)
        status, payload = api_server._batch_prediction_payload(files=files, fields=fields)
    except ValueError as exc:
        payload = {"ok": False, "error": str(exc)}
        status = HTTPStatus.BAD_REQUEST
    except Exception as exc:  # pragma: no cover - defensive WSGI surface
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        status = HTTPStatus.INTERNAL_SERVER_ERROR
    return _json_response(start_response, status, payload)


def app(environ: dict[str, Any], start_response: StartResponse) -> Iterable[bytes]:
    api_server.ensure_runtime_dirs()
    method = str(environ.get("REQUEST_METHOD", "GET")).upper()
    path = unquote(str(environ.get("PATH_INFO") or "/"))

    if method == "OPTIONS":
        return _json_response(start_response, HTTPStatus.OK, {"ok": True})

    if method == "GET":
        if path.rstrip("/") == "/health":
            return _json_response(start_response, HTTPStatus.OK, api_server.health_payload())
        if path.startswith("/outputs/"):
            response = _serve_file(start_response, api_server.OUTPUT_DIR, path, prefix="/outputs")
            if response is not None:
                return response
        if path == "/" or path in {"/index.html", "/app.js", "/styles.css"}:
            response = _serve_file(start_response, api_server.WEB_ROOT, path)
            if response is not None:
                return response
        return _json_response(start_response, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    if method == "POST":
        normalized_path = path.rstrip("/")
        if normalized_path == "/predict":
            return _handle_prediction(environ, start_response, wants_html=True)
        if normalized_path in {"/api/predict", "/analyze"}:
            return _handle_prediction(environ, start_response, wants_html=False)
        if normalized_path == "/api/predict-batch":
            return _handle_batch_prediction(environ, start_response)

    return _json_response(start_response, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})


application = app
