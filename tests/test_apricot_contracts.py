from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import apricot.inference as inference
from apricot.inference import ModelNotFoundError, predict_image_bytes
from apricot.schema_router import SchemaRoute, route_image_schema
from apricot.synthetic import (
    CLASS_NAME,
    CLASS_NAMES,
    PLATE_SCHEMA_LABELS,
    PLATE_SCHEMA_REGISTRY,
    SCHEMA_CLEAN_DOTS,
    SCHEMA_MERGED_SNOWMAN,
    SCHEMA_MIXED_PLATE,
    SCHEMA_STREAK_LINES,
    STRESS_SPLITS,
    SYNTHETIC_SUITE_SPLITS,
    build_domain_config,
    generate_dataset,
    generate_plate,
    generate_synthetic_suite,
    hex_to_bgr,
)
from gold_experience import api_server
from scripts.evaluate_synthetic_robustness import build_summary_rows, expected_count
from scripts.train_yolo import (
    CLEAN_DOT_MODEL_OUTPUT,
    MERGED_COLONY_MODEL_OUTPUT,
    training_specialist_for_schema,
    validate_specialist_output,
    validate_training_dataset,
)
from gold_experience.wsgi import app as wsgi_app


def test_plate_schema_registry_contains_required_labels() -> None:
    assert PLATE_SCHEMA_LABELS == (
        "clean_dots",
        "merged_snowman",
        "streak_lines",
        "mixed_plate",
        "unknown",
    )
    assert set(PLATE_SCHEMA_REGISTRY) == set(PLATE_SCHEMA_LABELS)
    assert PLATE_SCHEMA_REGISTRY[SCHEMA_CLEAN_DOTS].label == SCHEMA_CLEAN_DOTS


@pytest.fixture()
def apricot_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    model_dir = tmp_path / "models"
    monkeypatch.setattr(api_server, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(api_server, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(api_server, "MODEL_DIR", model_dir)
    api_server.ensure_runtime_dirs()

    def call(
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
        request_body = body or b""
        environ = {
            "REQUEST_METHOD": method,
            "PATH_INFO": path,
            "QUERY_STRING": "",
            "SERVER_NAME": "localhost",
            "SERVER_PORT": "80",
            "wsgi.version": (1, 0),
            "wsgi.url_scheme": "http",
            "wsgi.input": BytesIO(request_body),
            "wsgi.errors": BytesIO(),
            "wsgi.multithread": False,
            "wsgi.multiprocess": False,
            "wsgi.run_once": False,
            "CONTENT_LENGTH": str(len(request_body)),
        }
        for name, value in (headers or {}).items():
            normalized = name.upper().replace("-", "_")
            if normalized == "CONTENT_TYPE":
                environ["CONTENT_TYPE"] = value
            else:
                environ[f"HTTP_{normalized}"] = value

        captured: dict[str, object] = {}

        def start_response(status: str, response_headers: list[tuple[str, str]]) -> None:
            captured["status"] = status
            captured["headers"] = response_headers

        response_body = b"".join(wsgi_app(environ, start_response))
        status_code = int(str(captured["status"]).split()[0])
        return status_code, response_body

    call.upload_dir = upload_dir
    call.output_dir = output_dir
    call.model_dir = model_dir
    return call


def _png_bytes(value: int = 192) -> bytes:
    image = np.full((64, 64, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _clean_dot_image() -> np.ndarray:
    image = np.full((180, 180, 3), 206, dtype=np.uint8)
    for y_pos in range(35, 155, 34):
        for x_pos in range(32, 158, 31):
            cv2.circle(image, (x_pos, y_pos), 5, (82, 82, 82), -1, lineType=cv2.LINE_AA)
    return image


def _merged_snowman_image() -> np.ndarray:
    image = np.full((180, 180, 3), 206, dtype=np.uint8)
    centers = [
        (50, 50),
        (60, 50),
        (95, 50),
        (105, 55),
        (50, 92),
        (61, 101),
        (100, 96),
        (111, 96),
        (62, 138),
        (75, 138),
        (124, 136),
        (136, 144),
    ]
    for center in centers:
        cv2.circle(image, center, 6, (82, 82, 82), -1, lineType=cv2.LINE_AA)
    return image


def _streak_line_image() -> np.ndarray:
    image = np.full((180, 180, 3), 206, dtype=np.uint8)
    cv2.line(image, (24, 48), (158, 56), (72, 72, 72), 7, lineType=cv2.LINE_AA)
    cv2.line(image, (38, 92), (148, 118), (76, 76, 76), 6, lineType=cv2.LINE_AA)
    cv2.line(image, (52, 148), (160, 138), (80, 80, 80), 5, lineType=cv2.LINE_AA)
    return image


def _multipart_file(
    filename: str,
    payload: bytes,
    content_type: str = "image/png",
    fields: dict[str, str] | None = None,
) -> tuple[bytes, str]:
    boundary = "apricot-test-boundary"
    body = b""
    for name, value in (fields or {}).items():
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode("utf-8")
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    body += payload
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def _multipart_files(
    files: list[tuple[str, str, str, bytes]],
    fields: dict[str, str] | None = None,
) -> tuple[bytes, str]:
    boundary = "apricot-test-boundary"
    body = b""
    for name, value in (fields or {}).items():
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode("utf-8")
    for field_name, filename, content_type, payload in files:
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        body += payload
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def test_home_route_serves_web_app(apricot_client) -> None:
    status, body = apricot_client("/")
    assert status == 200
    assert b"Apricot" in body


def test_health_route_reports_model_configuration(apricot_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "models" / "apricot_clean_dot_counter_v1.pt"
    monkeypatch.setenv("APRICOT_MODEL_PATH", str(model_path))

    status, body = apricot_client("/health")
    payload = json.loads(body)

    assert status == 200
    assert payload["status"] == "ok"
    assert payload["model_path"] == model_path.name
    assert payload["model_exists"] is False
    assert payload["model_version"] == "missing"
    assert payload["model_loaded"] is False
    assert payload["default_confidence"] == pytest.approx(0.25)
    assert payload["retention_mode"] == "ephemeral"
    assert payload["max_batch_images"] == api_server.MAX_BATCH_IMAGES
    assert payload["max_total_upload_bytes"] == api_server.MAX_TOTAL_UPLOAD_BYTES
    assert payload["supported_schemas"] == ["clean_dots", "merged_snowman"]
    assert payload["coming_soon_schemas"] == ["streak_lines"]
    assert payload["model_specialists"]["clean_dots"]["display_name"] == "Clean-dot counter"
    assert payload["model_specialists"]["clean_dots"]["default_path"] == "models/apricot_clean_dot_counter_v1.pt"
    assert payload["model_specialists"]["clean_dots"]["path"] == model_path.name
    assert payload["model_specialists"]["merged_snowman"]["display_name"] == "Merged-colony counter"
    assert (
        payload["model_specialists"]["merged_snowman"]["default_path"]
        == "models/apricot_merged_colony_counter_v1.pt"
    )
    assert payload["model_specialists"]["merged_snowman"]["path"] == "models/apricot_merged_colony_counter_v1.pt"
    assert payload["version"]


def test_upload_validation_rejects_bad_file_type(apricot_client) -> None:
    body, content_type = _multipart_file("notes.txt", b"not an image", content_type="text/plain")

    status, response_body = apricot_client(
        "/api/predict",
        method="POST",
        body=body,
        headers={"Content-Type": content_type},
    )
    payload = json.loads(response_body)

    assert status == 415
    assert payload["ok"] is False
    assert "Unsupported upload type" in payload["error"]


def test_corrupt_supported_image_returns_bad_request(apricot_client) -> None:
    body, content_type = _multipart_file("plate.png", b"not an image", content_type="image/png")

    status, response_body = apricot_client(
        "/api/predict",
        method="POST",
        body=body,
        headers={"Content-Type": content_type},
    )
    payload = json.loads(response_body)

    assert status == 400
    assert payload["ok"] is False
    assert "readable image" in payload["error"]


def test_missing_model_returns_service_error(apricot_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APRICOT_MODEL_PATH", str(tmp_path / "missing.pt"))
    body, content_type = _multipart_file("plate.png", _png_bytes(), fields={"confidence": "0.42"})

    status, response_body = apricot_client(
        "/api/predict",
        method="POST",
        body=body,
        headers={"Content-Type": content_type},
    )
    payload = json.loads(response_body)

    assert status == 503
    assert payload["ok"] is False
    assert "Model weights not found" in payload["error"]


def test_inference_route_can_be_faked_without_yolo_weights(
    apricot_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def __init__(self, annotated_image_path: Path | None = None) -> None:
            self.annotated_image_path = annotated_image_path
            self.count = 2
            self.confidence_threshold = 0.42

        def to_dict(self) -> dict[str, object]:
            return {
                "count": 2,
                "detections": [],
                "boxes": [],
                "confidence_scores": [],
                "confidence_threshold": 0.42,
                "threshold_used": 0.42,
                "iou_threshold": 0.7,
                "max_det": 1000,
                "model_path": "fake.pt",
                "model_version": "fake",
                "image": {"width": 64, "height": 64},
                "annotated_image_path": None,
            }

    def fake_predict_image_array(
        image: np.ndarray,
        *,
        confidence: float,
        iou: float,
        max_det: int,
        output_dir: Path | None,
        output_stem: str,
        model_path: str | Path | None = None,
    ) -> FakeResult:
        assert image.shape == (64, 64, 3)
        assert confidence == pytest.approx(0.42)
        assert output_dir is None
        assert model_path is not None
        assert Path(model_path).name == "apricot_clean_dot_counter_v1.pt"
        return FakeResult()

    monkeypatch.setattr(api_server, "predict_image_array", fake_predict_image_array)
    body, content_type = _multipart_file("plate.png", _png_bytes(), fields={"confidence": "0.42"})

    status, response_body = apricot_client(
        "/api/predict",
        method="POST",
        body=body,
        headers={"Content-Type": content_type},
    )
    payload = json.loads(response_body)

    assert status == 200
    assert payload["ok"] is True
    assert payload["count"] == 2
    assert payload["schema"] == "clean_dots"
    assert payload["selected_schema"] == "clean_dots"
    assert payload["selected_model"]["schema"] == "clean_dots"
    assert payload["selected_model"]["path"].endswith("apricot_clean_dot_counter_v1.pt")
    assert payload["confidence"] == payload["schema_confidence"]
    assert payload["route"]["schema"] == payload["schema"]
    assert payload["route"]["confidence"] == payload["confidence"]
    assert payload["route_metadata"]["component_count"] >= 0
    assert payload["annotated_image_url"] is None
    assert "annotated_image_base64" not in payload
    assert "annotated_image_data_url" not in payload
    assert not any(apricot_client.upload_dir.iterdir())
    assert not any(apricot_client.output_dir.iterdir())


def test_batch_endpoint_returns_per_image_counts(apricot_client, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResult:
        def __init__(self, count: int, confidence: float) -> None:
            self.annotated_image_path = None
            self.annotated_image_bytes = _png_bytes(160 + count)
            self.count = count
            self.confidence_threshold = confidence

        def to_dict(self) -> dict[str, object]:
            return {
                "count": self.count,
                "detections": [],
                "boxes": [],
                "confidence_scores": [],
                "confidence_threshold": self.confidence_threshold,
                "threshold_used": self.confidence_threshold,
                "iou_threshold": 0.7,
                "max_det": 1000,
                "model_path": "fake.pt",
                "model_version": "fake",
                "image": {"width": 64, "height": 64},
                "annotated_image_path": None,
            }

    calls: list[float] = []

    def fake_predict_image_array(
        image: np.ndarray,
        *,
        confidence: float,
        iou: float,
        max_det: int,
        output_dir: Path | None,
        output_stem: str,
        model_path: str | Path | None = None,
    ) -> FakeResult:
        assert output_dir is None
        calls.append(confidence)
        return FakeResult(len(calls), confidence)

    monkeypatch.setattr(api_server, "predict_image_array", fake_predict_image_array)
    body, content_type = _multipart_files(
        [
            ("images", "plate1.png", "image/png", _png_bytes()),
            ("images", "plate2.png", "image/png", _png_bytes(180)),
        ],
        fields={"confidence": "0.25"},
    )

    status, response_body = apricot_client(
        "/api/predict-batch",
        method="POST",
        body=body,
        headers={"Content-Type": content_type},
    )
    payload = json.loads(response_body)

    assert status == 200
    assert payload["ok"] is True
    assert payload["images_processed"] == 2
    assert payload["images_failed"] == 0
    assert payload["count_total"] == 3
    assert [result["count"] for result in payload["results"]] == [1, 2]
    assert all(result["annotated_image_base64"] for result in payload["results"])
    assert all(result["annotated_image_url"] is None for result in payload["results"])
    assert calls == [pytest.approx(0.25), pytest.approx(0.25)]
    assert not any(apricot_client.upload_dir.iterdir())
    assert not any(apricot_client.output_dir.iterdir())


def test_batch_endpoint_returns_per_file_errors(apricot_client, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResult:
        annotated_image_path = None
        annotated_image_bytes = _png_bytes()
        count = 4
        confidence_threshold = 0.25

        def to_dict(self) -> dict[str, object]:
            return {
                "count": self.count,
                "detections": [],
                "boxes": [],
                "confidence_scores": [],
                "confidence_threshold": 0.25,
                "threshold_used": 0.25,
                "iou_threshold": 0.7,
                "max_det": 1000,
                "model_path": "fake.pt",
                "model_version": "fake",
                "image": {"width": 64, "height": 64},
                "annotated_image_path": None,
            }

    def fake_predict_image_array(image: np.ndarray, **kwargs: object) -> FakeResult:
        assert image.shape == (64, 64, 3)
        return FakeResult()

    monkeypatch.setattr(api_server, "predict_image_array", fake_predict_image_array)
    body, content_type = _multipart_files(
        [
            ("images", "plate.png", "image/png", _png_bytes()),
            ("images", "notes.txt", "text/plain", b"not an image"),
            ("images", "corrupt.png", "image/png", b"not an image"),
        ],
        fields={"confidence": "0.25"},
    )

    status, response_body = apricot_client(
        "/api/predict-batch",
        method="POST",
        body=body,
        headers={"Content-Type": content_type},
    )
    payload = json.loads(response_body)

    assert status == 200
    assert payload["ok"] is True
    assert payload["images_processed"] == 1
    assert payload["images_failed"] == 2
    assert payload["count_total"] == 4
    assert {error["filename"] for error in payload["errors"]} == {"notes.txt", "corrupt.png"}
    assert any("Unsupported file type" in error["error"] for error in payload["errors"])
    assert any("readable image" in error["error"] for error in payload["errors"])


def test_batch_endpoint_rejects_too_many_files(apricot_client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_server, "MAX_BATCH_IMAGES", 1)
    body, content_type = _multipart_files(
        [
            ("images", "plate1.png", "image/png", _png_bytes()),
            ("images", "plate2.png", "image/png", _png_bytes()),
        ]
    )

    status, response_body = apricot_client(
        "/api/predict-batch",
        method="POST",
        body=body,
        headers={"Content-Type": content_type},
    )
    payload = json.loads(response_body)

    assert status == 400
    assert payload["ok"] is False
    assert "Too many images" in payload["error"]


def test_batch_missing_model_returns_service_error(apricot_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APRICOT_MODEL_PATH", str(tmp_path / "missing.pt"))
    body, content_type = _multipart_files(
        [
            ("images", "plate1.png", "image/png", _png_bytes()),
            ("images", "plate2.png", "image/png", _png_bytes()),
        ]
    )

    status, response_body = apricot_client(
        "/api/predict-batch",
        method="POST",
        body=body,
        headers={"Content-Type": content_type},
    )
    payload = json.loads(response_body)

    assert status == 503
    assert payload["ok"] is False
    assert payload["images_processed"] == 0
    assert payload["images_failed"] == 2
    assert "Model weights not found" in payload["error"]


def test_merged_snowman_route_uses_merged_specialist_model(
    apricot_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        annotated_image_path = None
        annotated_image_bytes = _png_bytes()
        count = 3
        confidence_threshold = 0.42

        def to_dict(self) -> dict[str, object]:
            return {
                "count": 3,
                "detections": [],
                "boxes": [],
                "confidence_scores": [],
                "confidence_threshold": 0.42,
                "threshold_used": 0.42,
                "iou_threshold": 0.7,
                "max_det": 1000,
                "model_path": "fake-merged.pt",
                "model_version": "fake",
                "image": {"width": 64, "height": 64},
                "annotated_image_path": None,
            }

    def fake_route(_image: np.ndarray) -> SchemaRoute:
        return SchemaRoute(
            schema=SCHEMA_MERGED_SNOWMAN,
            confidence=0.83,
            metadata={
                "component_count": 4,
                "merged_component_count": 3,
                "scores": {
                    SCHEMA_CLEAN_DOTS: 0.41,
                    SCHEMA_MERGED_SNOWMAN: 0.84,
                    SCHEMA_STREAK_LINES: 0.12,
                },
            },
        )

    def fake_predict_image_array(
        image: np.ndarray,
        *,
        confidence: float,
        iou: float,
        max_det: int,
        output_dir: Path | None,
        output_stem: str,
        model_path: str | Path | None = None,
    ) -> FakeResult:
        assert image.shape == (64, 64, 3)
        assert confidence == pytest.approx(0.42)
        assert output_dir is None
        assert output_stem == "annotated"
        assert model_path is not None
        assert Path(model_path).name == "apricot_merged_colony_counter_v1.pt"
        return FakeResult()

    monkeypatch.setattr(api_server, "route_image_schema", fake_route)
    monkeypatch.setattr(api_server, "predict_image_array", fake_predict_image_array)
    body, content_type = _multipart_file("plate.png", _png_bytes(), fields={"confidence": "0.42"})

    status, response_body = apricot_client(
        "/api/predict",
        method="POST",
        body=body,
        headers={"Content-Type": content_type},
    )
    payload = json.loads(response_body)

    assert status == 200
    assert payload["ok"] is True
    assert payload["schema"] == SCHEMA_MERGED_SNOWMAN
    assert payload["selected_schema"] == SCHEMA_MERGED_SNOWMAN
    assert payload["selected_model"]["schema"] == SCHEMA_MERGED_SNOWMAN
    assert payload["selected_model"]["path"].endswith("apricot_merged_colony_counter_v1.pt")
    assert payload["model"]["schema"] == SCHEMA_MERGED_SNOWMAN
    assert payload["route"]["metadata"]["merged_component_count"] == 3


def test_schema_router_detects_clean_dot_images() -> None:
    route = route_image_schema(_clean_dot_image())

    assert route.schema == "clean_dots"
    assert route.confidence >= 0.55
    assert route.metadata["round_component_count"] >= 12
    assert route.metadata["elongated_component_count"] == 0
    assert route.metadata["scores"]["merged_snowman"] < route.metadata["scores"]["clean_dots"]


def test_schema_router_detects_merged_snowman_images() -> None:
    route = route_image_schema(_merged_snowman_image())

    assert route.schema == "merged_snowman"
    assert route.confidence >= 0.55
    assert route.metadata["merged_component_count"] >= 4
    assert route.metadata["scores"]["merged_snowman"] > route.metadata["scores"]["clean_dots"]
    assert route.metadata["scores"]["merged_snowman"] > route.metadata["scores"]["streak_lines"]


def test_schema_router_detects_streak_line_images() -> None:
    route = route_image_schema(_streak_line_image())

    assert route.schema == "streak_lines"
    assert route.confidence >= 0.55
    assert route.metadata["elongated_component_count"] >= 2
    assert route.metadata["line_like_area_ratio"] > 0.1
    assert route.metadata["scores"]["streak_lines"] > route.metadata["scores"]["merged_snowman"]


def test_prediction_function_can_use_fake_yolo_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBoxes:
        xyxy = np.array([[5.0, 6.0, 20.0, 24.0]], dtype=float)
        conf = np.array([0.91], dtype=float)
        cls = np.array([0], dtype=float)

    class FakeResult:
        boxes = FakeBoxes()
        names = {0: "colony"}

    class FakeModel:
        def predict(self, **kwargs: object) -> list[FakeResult]:
            self.kwargs = kwargs
            return [FakeResult()]

    fake_model = FakeModel()
    model_path = tmp_path / "fake.pt"
    model_path.write_bytes(b"fake weights")
    image_path = tmp_path / "plate.png"
    image_path.write_bytes(_png_bytes())
    monkeypatch.setattr(inference, "load_model", lambda _model_path: fake_model)

    result = inference.predict_colonies(
        image_path,
        conf=0.15,
        model_path=model_path,
        output_dir=tmp_path,
        output_stem="annotated",
    )

    assert result["count"] == 1
    assert result["boxes"][0]["class_name"] == "colony"
    assert result["confidence_scores"][0] == pytest.approx(0.91)
    assert fake_model.kwargs["conf"] == pytest.approx(0.15)


def test_synthetic_generator_writes_dataset_contracts(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic"
    summary = generate_dataset(
        output_dir=output_dir,
        plates=4,
        image_size=256,
        train_ratio=0.75,
        seed=3,
        size_mode="small",
        species="s_cerevisiae",
        medium="YPD",
        colony_count_ranges={"small": (2, 4)},
    )

    assert summary["plates"] == 4
    assert summary["class_name"] == CLASS_NAME
    assert len(list((output_dir / "images" / "train").glob("*.jpg"))) == 3
    assert len(list((output_dir / "images" / "val").glob("*.jpg"))) == 1

    dataset_yaml = (output_dir / "dataset.yaml").read_text(encoding="utf-8")
    assert f"path: {output_dir.resolve()}" in dataset_yaml
    assert "train: images/train" in dataset_yaml
    assert "val: images/val" in dataset_yaml
    assert "nc: 1" in dataset_yaml
    assert f'names: ["{CLASS_NAME}"]' in dataset_yaml

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator_version"]
    assert manifest["schema"] == SCHEMA_CLEAN_DOTS
    assert manifest["schema_labels"] == list(PLATE_SCHEMA_LABELS)
    assert manifest["plate_schema_registry"][SCHEMA_CLEAN_DOTS]["label"] == SCHEMA_CLEAN_DOTS
    assert manifest["random_seed"] == 3
    assert manifest["number_of_images"] == 4
    assert manifest["class_names"] == CLASS_NAMES
    assert manifest["species_profile"]["id"] == "s_cerevisiae"
    assert manifest["medium_profile"]["id"] == "YPD"
    assert manifest["train_val_split"] == {"train_ratio": 0.75, "train": 3, "val": 1}
    assert len(manifest["images"]) == 4
    for record in manifest["images"]:
        assert set(record) >= {
            "image",
            "label",
            "split",
            "stem",
            "schema",
            "species",
            "medium",
            "colony_count",
            "requested_colonies",
            "placed_colonies",
            "colonies",
            "generator_parameters",
        }
        assert record["schema"] == SCHEMA_CLEAN_DOTS
        assert record["schema"] not in {SCHEMA_MERGED_SNOWMAN, SCHEMA_STREAK_LINES, SCHEMA_MIXED_PLATE}
        assert record["species"] == "s_cerevisiae"
        assert record["medium"] == "YPD"
        assert record["colony_count"] == record["placed_colonies"]
        assert record["generator_parameters"]["schema"] == SCHEMA_CLEAN_DOTS
        assert record["generator_parameters"]["species"] == "s_cerevisiae"
        assert record["generator_parameters"]["medium"] == "YPD"
        assert record["generator_parameters"]["colony_count"] == record["colony_count"]
        assert record["generator_parameters"]["random_seed"] == 3
        assert (output_dir / record["image"]).is_file()
        assert (output_dir / record["label"]).is_file()


def test_clean_dot_dataset_schema_is_not_mixed_with_snowman_or_streak_data(tmp_path: Path) -> None:
    output_dir = tmp_path / "clean_dots"
    generate_dataset(
        output_dir=output_dir,
        plates=5,
        image_size=256,
        train_ratio=0.8,
        seed=23,
        size_mode="small",
        colony_count_ranges={"small": (2, 3)},
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    record_schemas = {record["schema"] for record in manifest["images"]}
    parameter_schemas = {record["generator_parameters"]["schema"] for record in manifest["images"]}

    assert manifest["schema"] == SCHEMA_CLEAN_DOTS
    assert record_schemas == {SCHEMA_CLEAN_DOTS}
    assert parameter_schemas == {SCHEMA_CLEAN_DOTS}
    assert record_schemas.isdisjoint({SCHEMA_MERGED_SNOWMAN, SCHEMA_STREAK_LINES})


def test_merged_snowman_plate_mode_writes_touching_instances() -> None:
    config = build_domain_config(species="s_cerevisiae", medium="YPD", image_size=320, collision_margin=8)
    expected_shapes = {2: "snowman", 3: "tri_lobed", 4: "small_clump"}

    for colony_count, expected_shape in expected_shapes.items():
        plate = generate_plate(
            colony_count=colony_count,
            size_mode="small",
            schema=SCHEMA_MERGED_SNOWMAN,
            rng=np.random.default_rng(100 + colony_count),
            config=config,
        )

        assert plate.metadata["schema"] == SCHEMA_MERGED_SNOWMAN
        assert plate.metadata["generator_parameters"]["schema"] == SCHEMA_MERGED_SNOWMAN
        assert len(plate.labels) == colony_count
        assert len(plate.metadata["colonies"]) == colony_count
        assert plate.metadata["merged_clusters"][0]["shape"] == expected_shape
        assert plate.metadata["merged_clusters"][0]["colony_count"] == colony_count
        assert 2 <= plate.metadata["merged_clusters"][0]["colony_count"] <= 4

        colonies = plate.metadata["colonies"]
        assert {colony["cluster_id"] for colony in colonies} == {0}
        assert {colony["cluster_shape"] for colony in colonies} == {expected_shape}
        touching_pairs = 0
        for first_index, first in enumerate(colonies):
            for second in colonies[first_index + 1 :]:
                distance = float(np.hypot(first["x"] - second["x"], first["y"] - second["y"]))
                if distance < first["radius"] + second["radius"]:
                    touching_pairs += 1
        assert touching_pairs >= colony_count - 1


def test_merged_snowman_dataset_is_labeled_separately_from_clean_dots(tmp_path: Path) -> None:
    output_dir = tmp_path / "merged_snowman"
    summary = generate_dataset(
        output_dir=output_dir,
        plates=3,
        image_size=384,
        train_ratio=0.67,
        seed=31,
        size_mode="small",
        schema=SCHEMA_MERGED_SNOWMAN,
        colony_count_ranges={"small": (9, 9)},
    )

    assert summary["schema"] == SCHEMA_MERGED_SNOWMAN

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == SCHEMA_MERGED_SNOWMAN
    assert manifest["schema"] != SCHEMA_CLEAN_DOTS
    assert manifest["schema_labels"] == list(PLATE_SCHEMA_LABELS)
    assert manifest["plate_schema_registry"][SCHEMA_MERGED_SNOWMAN]["label"] == SCHEMA_MERGED_SNOWMAN

    for record in manifest["images"]:
        label_path = output_dir / record["label"]
        label_count = sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())

        assert record["schema"] == SCHEMA_MERGED_SNOWMAN
        assert record["schema"] != SCHEMA_CLEAN_DOTS
        assert record["generator_parameters"]["schema"] == SCHEMA_MERGED_SNOWMAN
        assert record["colony_count"] == label_count
        assert record["colony_count"] == len(record["colonies"])
        assert record["placed_clusters"] == len(record["merged_clusters"])
        assert record["merged_clusters"]
        assert all(2 <= cluster["colony_count"] <= 4 for cluster in record["merged_clusters"])

    with pytest.raises(SystemExit, match="Refusing to train the clean-dot model"):
        validate_training_dataset(output_dir / "dataset.yaml")

    inspection = validate_training_dataset(
        output_dir / "dataset.yaml",
        training_schema=SCHEMA_MERGED_SNOWMAN,
    )
    assert inspection.schemas == (SCHEMA_MERGED_SNOWMAN,)


def _write_training_dataset(dataset_dir: Path, manifest: dict[str, object]) -> Path:
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset_yaml = dataset_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_dir}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                'names: ["yeast_colony"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def test_train_yolo_accepts_clean_dot_schema(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "clean_dots"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"schema": SCHEMA_CLEAN_DOTS, "images": [{"schema": SCHEMA_CLEAN_DOTS}]}),
        encoding="utf-8",
    )
    dataset_yaml = dataset_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_dir}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                'names: ["yeast_colony"]',
                "",
            ]
        ),
        encoding="utf-8",
    )

    inspection = validate_training_dataset(dataset_yaml)

    assert inspection.schemas == (SCHEMA_CLEAN_DOTS,)


def test_train_yolo_accepts_merged_snowman_specialist_schema(tmp_path: Path) -> None:
    dataset_yaml = _write_training_dataset(
        tmp_path / "merged_snowman",
        {"schema": SCHEMA_MERGED_SNOWMAN, "images": [{"schema": SCHEMA_MERGED_SNOWMAN}]},
    )

    inspection = validate_training_dataset(dataset_yaml, training_schema=SCHEMA_MERGED_SNOWMAN)
    specialist = training_specialist_for_schema(SCHEMA_MERGED_SNOWMAN)

    assert inspection.schemas == (SCHEMA_MERGED_SNOWMAN,)
    assert specialist.display_name == "merged-colony counter"
    assert specialist.output_path == MERGED_COLONY_MODEL_OUTPUT
    assert specialist.output_path.name == "apricot_merged_colony_counter_v1.pt"
    assert specialist.output_path != CLEAN_DOT_MODEL_OUTPUT


def test_train_yolo_rejects_snowman_or_streak_schema_by_default(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "clean_dots"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"schema": SCHEMA_CLEAN_DOTS, "images": [{"schema": SCHEMA_MERGED_SNOWMAN}]}),
        encoding="utf-8",
    )
    dataset_yaml = dataset_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_dir}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                'names: ["yeast_colony"]',
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Refusing to train the clean-dot model"):
        validate_training_dataset(dataset_yaml)


def test_train_yolo_rejects_clean_dot_data_for_merged_specialist(tmp_path: Path) -> None:
    dataset_yaml = _write_training_dataset(
        tmp_path / "clean_dots",
        {"schema": SCHEMA_CLEAN_DOTS, "images": [{"schema": SCHEMA_CLEAN_DOTS}]},
    )

    with pytest.raises(SystemExit, match="Refusing to train the merged-colony counter"):
        validate_training_dataset(dataset_yaml, training_schema=SCHEMA_MERGED_SNOWMAN)


def test_train_yolo_protects_specialist_output_paths() -> None:
    validate_specialist_output(SCHEMA_MERGED_SNOWMAN, MERGED_COLONY_MODEL_OUTPUT)
    validate_specialist_output(SCHEMA_CLEAN_DOTS, CLEAN_DOT_MODEL_OUTPUT)

    with pytest.raises(SystemExit, match="clean-dot counter specialist output"):
        validate_specialist_output(SCHEMA_MERGED_SNOWMAN, CLEAN_DOT_MODEL_OUTPUT)

    with pytest.raises(SystemExit, match="merged-colony counter specialist output"):
        validate_specialist_output(SCHEMA_CLEAN_DOTS, MERGED_COLONY_MODEL_OUTPUT)

    validate_specialist_output(
        SCHEMA_MERGED_SNOWMAN,
        CLEAN_DOT_MODEL_OUTPUT,
        allow_specialist_output_mismatch=True,
    )


def test_train_yolo_allows_explicit_include_schema_override(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "clean_dots"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"schema": SCHEMA_CLEAN_DOTS, "images": [{"schema": SCHEMA_STREAK_LINES}]}),
        encoding="utf-8",
    )
    dataset_yaml = dataset_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_dir}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                'names: ["yeast_colony"]',
                "",
            ]
        ),
        encoding="utf-8",
    )

    inspection = validate_training_dataset(dataset_yaml, include_schemas=(SCHEMA_STREAK_LINES,))

    assert inspection.schemas == (SCHEMA_CLEAN_DOTS, SCHEMA_STREAK_LINES)


def test_synthetic_suite_writes_named_split_contracts(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    summary = generate_synthetic_suite(
        output_dir=suite_dir,
        train_plates=2,
        val_plates=1,
        stress_plates=1,
        image_size=512,
        seed=11,
    )

    assert Path(summary["dataset_yaml"]).is_file()
    root_yaml = (suite_dir / "dataset.yaml").read_text(encoding="utf-8")
    assert "train: train_standard/images" in root_yaml
    assert "val: val_standard/images" in root_yaml

    suite_manifest = json.loads((suite_dir / "suite_manifest.json").read_text(encoding="utf-8"))
    assert suite_manifest["schema"] == SCHEMA_CLEAN_DOTS
    assert suite_manifest["protocol"] == "synthetic_only_three_tier"
    assert suite_manifest["schema_labels"] == list(PLATE_SCHEMA_LABELS)
    assert suite_manifest["plate_schema_registry"][SCHEMA_STREAK_LINES]["label"] == SCHEMA_STREAK_LINES
    assert suite_manifest["tiers"]["ood_synthetic_stress"] == list(STRESS_SPLITS)

    for split_name in SYNTHETIC_SUITE_SPLITS:
        split_dir = suite_dir / split_name
        assert (split_dir / "images").is_dir()
        assert (split_dir / "labels").is_dir()
        assert (split_dir / "dataset.yaml").is_file()
        manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["split"] == split_name
        assert manifest["schema"] == SCHEMA_CLEAN_DOTS
        assert manifest["schema_labels"] == list(PLATE_SCHEMA_LABELS)
        assert manifest["expected_counts"]
        expected_images = 2 if split_name == "train_standard" else 1
        assert manifest["number_of_images"] == expected_images
        assert len(manifest["images"]) == expected_images
        for record in manifest["images"]:
            label_path = split_dir / record["label"]
            label_count = sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
            assert record["schema"] == SCHEMA_CLEAN_DOTS
            assert record["schema"] not in {SCHEMA_MERGED_SNOWMAN, SCHEMA_STREAK_LINES, SCHEMA_MIXED_PLATE}
            assert record["colony_count"] == label_count
            assert record["generator_parameters"]["schema"] == SCHEMA_CLEAN_DOTS
            assert record["generator_parameters"]["colony_count"] == label_count
            assert record["expected_colony_count"] == label_count
            assert expected_count(record, split_dir) == label_count


def test_robustness_summary_marks_best_threshold() -> None:
    rows = [
        {
            "split": "test_lighting_shift",
            "tier": "ood_synthetic_stress",
            "threshold": "0.10",
            "absolute_count_error": 4,
            "percent_count_error": "20.0",
            "signed_count_error": 4,
        },
        {
            "split": "test_lighting_shift",
            "tier": "ood_synthetic_stress",
            "threshold": "0.10",
            "absolute_count_error": 2,
            "percent_count_error": "10.0",
            "signed_count_error": 2,
        },
        {
            "split": "test_lighting_shift",
            "tier": "ood_synthetic_stress",
            "threshold": "0.25",
            "absolute_count_error": 1,
            "percent_count_error": "5.0",
            "signed_count_error": -1,
        },
        {
            "split": "test_lighting_shift",
            "tier": "ood_synthetic_stress",
            "threshold": "0.25",
            "absolute_count_error": 1,
            "percent_count_error": "5.0",
            "signed_count_error": 1,
        },
    ]

    summary = build_summary_rows(rows)
    best_rows = [row for row in summary if row["is_best_threshold"] == "true"]
    assert len(best_rows) == 1
    assert best_rows[0]["threshold"] == "0.25"
    assert best_rows[0]["best_threshold_for_split"] == "0.25"


def test_generated_plate_pixels_follow_dish_contract() -> None:
    config = build_domain_config(species="s_cerevisiae", medium="YPD", image_size=256, collision_margin=7)
    plate = generate_plate(colony_count=5, size_mode="small", rng=np.random.default_rng(11), config=config)

    assert tuple(int(value) for value in plate.image[0, 0]) == (0, 0, 0)
    assert tuple(int(value) for value in plate.image[0, -1]) == (0, 0, 0)
    assert tuple(int(value) for value in plate.image[-1, 0]) == (0, 0, 0)
    assert tuple(int(value) for value in plate.image[-1, -1]) == (0, 0, 0)
    assert np.any(plate.image[config.image_size // 2, config.image_size // 2] != 0)


def test_synthetic_labels_are_normalized_inside_dish_and_non_overlapping(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic"
    generate_dataset(
        output_dir=output_dir,
        plates=6,
        image_size=256,
        train_ratio=0.8,
        seed=7,
        size_mode="small",
        species="p_pastoris",
        medium="unknown_future_medium",
        collision_margin=6,
        colony_count_ranges={"small": (3, 5)},
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["medium_profile"]["id"] == "unknown_future_medium"
    assert manifest["medium_profile"]["base_profile"] == "generic_dark_agar"
    assert manifest["no_overlap_margin"] == 6

    for record in manifest["images"]:
        label_path = output_dir / record["label"]
        assert label_path.is_file()
        labels = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert labels

        dish = record["dish"]
        image_size = record["image_size"]
        for parts in labels:
            assert len(parts) == 5
            assert parts[0] == "0"
            normalized = [float(value) for value in parts[1:]]
            assert all(0.0 <= value <= 1.0 for value in normalized)
            x_pos = normalized[0] * image_size
            y_pos = normalized[1] * image_size
            distance_to_center = float(np.hypot(x_pos - dish["x"], y_pos - dish["y"]))
            assert distance_to_center <= dish["radius"]

        colonies = record["colonies"]
        for index, colony in enumerate(colonies):
            for other in colonies[index + 1 :]:
                distance = float(np.hypot(colony["x"] - other["x"], colony["y"] - other["y"]))
                min_distance = colony["radius"] + other["radius"] + manifest["no_overlap_margin"]
                assert distance >= min_distance


def test_hex_to_bgr_uses_opencv_channel_order() -> None:
    assert hex_to_bgr("#E4DCC6") == (198, 220, 228)


def test_predict_image_bytes_reports_missing_model_before_yolo_load(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="Model weights not found"):
        predict_image_bytes(_png_bytes(), model_path=tmp_path / "missing.pt")
