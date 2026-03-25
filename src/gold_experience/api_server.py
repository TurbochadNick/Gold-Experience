from __future__ import annotations

from email.parser import BytesParser
from email.policy import default
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from .frontend_payload import build_frontend_payload
from .pipeline import ColonyCounterPipeline

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")


def _parse_multipart_payload(body: bytes, content_type: str) -> tuple[bytes | None, str | None]:
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\n\r\n" + body
    )
    if not message.is_multipart():
        return None, None

    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        field_name = part.get_param("name", header="Content-Disposition")
        if field_name not in {"file", "image", "upload"}:
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        return payload, part.get_filename()

    return None, None


class GoldExperienceRequestHandler(BaseHTTPRequestHandler):
    pipeline = ColonyCounterPipeline()
    server_version = "GoldExperienceHTTP/0.1"

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

    def _parse_upload(self) -> tuple[bytes | None, str | None]:
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            return None, None

        payload = self.rfile.read(content_length)
        if "multipart/form-data" in content_type:
            return _parse_multipart_payload(payload, content_type)
        if content_type.startswith("image/"):
            return payload, self.headers.get("X-Filename")
        return None, None

    def _serve_static(self, request_path: str) -> bool:
        relative_path = request_path.lstrip("/") or "index.html"
        if relative_path.endswith("/"):
            relative_path = f"{relative_path}index.html"

        candidate = (WEB_ROOT / relative_path).resolve()
        if WEB_ROOT.resolve() not in candidate.parents and candidate != WEB_ROOT.resolve():
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
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "engine": "gold-experience-v1",
                },
            )
            return

        if path == "/" or path in {"/index.html", "/app.js", "/styles.css"}:
            if self._serve_static(path):
                return

        self._send_json(
            HTTPStatus.NOT_FOUND,
            {"ok": False, "error": "Not found"},
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/analyze":
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"ok": False, "error": "Not found"},
            )
            return

        try:
            file_bytes, filename = self._parse_upload()
            if not file_bytes:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": "Expected multipart form-data with a file field."},
                )
                return

            encoded = np.frombuffer(file_bytes, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": "Uploaded file is not a readable image."},
                )
                return

            result = self.pipeline.run(image)
            payload = build_frontend_payload(result=result, image_shape=image.shape, filename=filename)
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "analysis": payload,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive API surface
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            )


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), GoldExperienceRequestHandler)
    try:
        print(f"Gold Experience web app listening on http://{host}:{port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
