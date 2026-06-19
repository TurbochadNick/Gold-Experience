from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apricot.plate_schema import SCHEMA_CLEAN_DOTS, SCHEMA_MERGED_SNOWMAN, SCHEMA_STREAK_LINES  # noqa: E402


DEFAULT_TRAINING_SCHEMA = SCHEMA_CLEAN_DOTS
BLOCKED_CLEAN_DOT_SCHEMAS = {SCHEMA_MERGED_SNOWMAN, SCHEMA_STREAK_LINES}
BLOCKED_CLEAN_DOT_MARKERS = {"snowman", "snowmen", "streak", "streaks"}
SCHEMA_KEYS = {"schema", "dataset_schema"}
SCHEMA_LIST_KEYS = {"schemas", "dataset_schemas"}


@dataclass(frozen=True)
class DatasetInspection:
    dataset_root: Path
    data_paths: tuple[Path, ...]
    manifest_paths: tuple[Path, ...]
    schemas: tuple[str, ...]


@dataclass(frozen=True)
class TrainingSpecialist:
    schema: str
    display_name: str
    run_name: str
    output_path: Path


CLEAN_DOT_MODEL_OUTPUT = Path("models/apricot_clean_dot_counter_v1.pt")
MERGED_COLONY_MODEL_OUTPUT = Path("models/apricot_merged_colony_counter_v1.pt")
TRAINING_SPECIALISTS = {
    SCHEMA_CLEAN_DOTS: TrainingSpecialist(
        schema=SCHEMA_CLEAN_DOTS,
        display_name="clean-dot counter",
        run_name="apricot_clean_dot_counter_v1",
        output_path=CLEAN_DOT_MODEL_OUTPUT,
    ),
    SCHEMA_MERGED_SNOWMAN: TrainingSpecialist(
        schema=SCHEMA_MERGED_SNOWMAN,
        display_name="merged-colony counter",
        run_name="apricot_merged_colony_counter_v1",
        output_path=MERGED_COLONY_MODEL_OUTPUT,
    ),
}


def _normalize_schema(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _normalize_schema_values(values: tuple[str, ...] | list[str]) -> set[str]:
    schemas: set[str] = set()
    for value in values:
        for item in str(value).split(","):
            normalized = _normalize_schema(item)
            if normalized:
                schemas.add(normalized)
    return schemas


def training_specialist_for_schema(schema: str) -> TrainingSpecialist:
    normalized = _normalize_schema(schema)
    try:
        return TRAINING_SPECIALISTS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(TRAINING_SPECIALISTS))
        raise SystemExit(f"Unsupported Apricot training schema {schema!r}; supported schemas: {supported}.") from exc


def _resolve_for_compare(path: Path) -> Path:
    return path.expanduser().resolve()


def validate_specialist_output(
    training_schema: str,
    output: Path,
    *,
    allow_specialist_output_mismatch: bool = False,
) -> None:
    if allow_specialist_output_mismatch:
        return

    specialist = training_specialist_for_schema(training_schema)
    output_path = _resolve_for_compare(output)
    protected_outputs = {
        schema: _resolve_for_compare(config.output_path) for schema, config in TRAINING_SPECIALISTS.items()
    }

    for schema, protected_output in protected_outputs.items():
        if schema != specialist.schema and output_path == protected_output:
            protected_name = TRAINING_SPECIALISTS[schema].display_name
            raise SystemExit(
                f"Refusing to write {specialist.display_name} weights to the {protected_name} specialist output "
                f"({TRAINING_SPECIALISTS[schema].output_path}). Pass --allow-specialist-output-mismatch only for "
                "an intentional manual override."
            )


def _parse_dataset_yaml(dataset_yaml: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in dataset_yaml.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#") or raw_line[0].isspace():
            continue
        key, separator, value = raw_line.partition(":")
        if not separator:
            continue
        parsed[key.strip()] = value.strip().strip("\"'")
    return parsed


def _yaml_path_values(raw_value: str | None) -> tuple[str, ...]:
    if not raw_value:
        return ()
    value = raw_value.strip()
    if value.startswith("[") and value.endswith("]"):
        return tuple(item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip())
    return (value.strip("\"'"),)


def _resolve_dataset_root(dataset_yaml: Path, parsed_yaml: dict[str, str]) -> Path:
    raw_root = parsed_yaml.get("path")
    if not raw_root:
        return dataset_yaml.parent.resolve()
    dataset_root = Path(raw_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = dataset_yaml.parent / dataset_root
    return dataset_root.resolve()


def _resolve_data_paths(dataset_root: Path, parsed_yaml: dict[str, str]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for key in ("train", "val", "test"):
        for raw_path in _yaml_path_values(parsed_yaml.get(key)):
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = dataset_root / path
            paths.append(path.resolve())
    if not paths:
        paths.append(dataset_root)
    return tuple(dict.fromkeys(paths))


def _candidate_manifest_paths(dataset_yaml: Path, dataset_root: Path, data_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for root in (dataset_yaml.parent.resolve(), dataset_root):
        candidates.extend([root / "manifest.json", root / "suite_manifest.json"])

    stop_dirs = {dataset_yaml.parent.resolve(), dataset_root}
    for data_path in data_paths:
        current = data_path
        for _ in range(6):
            candidates.extend([current / "manifest.json", current / "suite_manifest.json"])
            if current in stop_dirs or current == current.parent:
                break
            current = current.parent

    return tuple(dict.fromkeys(path for path in candidates if path.is_file()))


def _collect_schema_values(payload: object) -> set[str]:
    schemas: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized_key = _normalize_schema(key)
            if normalized_key in SCHEMA_KEYS:
                schemas.add(_normalize_schema(value))
            elif normalized_key in SCHEMA_LIST_KEYS and isinstance(value, list):
                schemas.update(_normalize_schema(item) for item in value)
            else:
                schemas.update(_collect_schema_values(value))
    elif isinstance(payload, list):
        for item in payload:
            schemas.update(_collect_schema_values(item))
    return {schema for schema in schemas if schema}


def _tokenize_marker_text(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower().replace("_", " "))
    return {token for token in normalized.split() if token}


def _find_blocked_markers(*items: object) -> set[str]:
    markers: set[str] = set()
    for item in items:
        if isinstance(item, Path):
            tokens = _tokenize_marker_text(str(item))
        elif isinstance(item, (list, tuple, set)):
            tokens = _tokenize_marker_text(" ".join(str(value) for value in item))
        else:
            tokens = _tokenize_marker_text(str(item))
        markers.update(tokens & BLOCKED_CLEAN_DOT_MARKERS)
    return markers


def _marker_is_explicitly_allowed(marker: str, explicitly_allowed: set[str]) -> bool:
    plural = f"{marker}s"
    for schema in explicitly_allowed:
        tokens = _tokenize_marker_text(schema)
        if marker in tokens or plural in tokens:
            return True
    return False


def inspect_training_dataset(dataset_yaml: Path) -> DatasetInspection:
    if not dataset_yaml.is_file():
        raise SystemExit(f"Dataset YAML not found: {dataset_yaml}")

    parsed_yaml = _parse_dataset_yaml(dataset_yaml)
    dataset_root = _resolve_dataset_root(dataset_yaml, parsed_yaml)
    data_paths = _resolve_data_paths(dataset_root, parsed_yaml)
    manifest_paths = _candidate_manifest_paths(dataset_yaml, dataset_root, data_paths)
    schemas: set[str] = set()

    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Could not read dataset manifest as JSON: {manifest_path}") from exc
        schemas.update(_collect_schema_values(manifest))

    return DatasetInspection(
        dataset_root=dataset_root,
        data_paths=data_paths,
        manifest_paths=manifest_paths,
        schemas=tuple(sorted(schemas)),
    )


def validate_training_dataset(
    dataset_yaml: Path,
    include_schemas: tuple[str, ...] = (),
    training_schema: str = DEFAULT_TRAINING_SCHEMA,
) -> DatasetInspection:
    inspection = inspect_training_dataset(dataset_yaml)
    specialist = training_specialist_for_schema(training_schema)
    target_schema = specialist.schema
    explicitly_allowed = _normalize_schema_values(include_schemas)
    allowed_schemas = {target_schema, *explicitly_allowed}

    if not inspection.manifest_paths:
        raise SystemExit(
            f"No manifest found for {dataset_yaml}. {specialist.display_name} training requires a manifest "
            f"declaring schema={target_schema}."
        )
    if not inspection.schemas:
        manifests = ", ".join(str(path) for path in inspection.manifest_paths)
        raise SystemExit(
            f"Dataset manifests do not declare a schema: {manifests}. {specialist.display_name} training requires "
            f"schema={target_schema}; regenerate the dataset or pass --include-schema intentionally."
        )

    missing_target_schema = target_schema not in inspection.schemas
    disallowed_schemas = set(inspection.schemas) - allowed_schemas
    blocked_schemas: set[str] = set()
    accidental_markers: set[str] = set()
    if target_schema == SCHEMA_CLEAN_DOTS:
        blocked_schemas = {
            schema
            for schema in inspection.schemas
            if schema in BLOCKED_CLEAN_DOT_SCHEMAS and schema not in explicitly_allowed
        }
        accidental_markers = {
            marker
            for marker in _find_blocked_markers(
                dataset_yaml,
                inspection.dataset_root,
                inspection.data_paths,
                inspection.schemas,
            )
            if not _marker_is_explicitly_allowed(marker, explicitly_allowed)
        }

    if missing_target_schema or disallowed_schemas or blocked_schemas or accidental_markers:
        details = []
        if missing_target_schema:
            details.append(f"missing target schema: {target_schema}")
        if disallowed_schemas:
            details.append(f"disallowed schema(s): {', '.join(sorted(disallowed_schemas))}")
        if blocked_schemas:
            details.append(f"blocked clean-dot contamination schema(s): {', '.join(sorted(blocked_schemas))}")
        if accidental_markers:
            details.append(f"suspicious snowman/streak marker(s): {', '.join(sorted(accidental_markers))}")
        if target_schema == SCHEMA_CLEAN_DOTS:
            prefix = "Refusing to train the clean-dot model because the dataset is not clean_dots-only "
        else:
            prefix = (
                f"Refusing to train the {specialist.display_name} because the dataset is not "
                f"{target_schema}-only "
            )
        raise SystemExit(
            prefix
            + f"({'; '.join(details)}). Pass --include-schema <schema> only when you intentionally want mixed-schema training."
        )

    explicitly_mixed = sorted(
        schema for schema in inspection.schemas if schema in explicitly_allowed and schema != target_schema
    )
    if explicitly_mixed:
        print(
            f"WARNING: {specialist.display_name} training includes explicitly allowed non-{target_schema} schema(s): "
            + ", ".join(explicitly_mixed),
            file=sys.stderr,
        )

    return inspection


def train_yolo(
    *,
    dataset_yaml: Path,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    patience: int,
    project: Path,
    name: str,
    output: Path,
    include_schemas: tuple[str, ...] = (),
    training_schema: str = DEFAULT_TRAINING_SCHEMA,
    allow_specialist_output_mismatch: bool = False,
) -> Path:
    validate_training_dataset(dataset_yaml, include_schemas=include_schemas, training_schema=training_schema)
    validate_specialist_output(
        training_schema,
        output,
        allow_specialist_output_mismatch=allow_specialist_output_mismatch,
    )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Install training dependencies with `pip install -e .` "
            "or `pip install ultralytics`, then rerun this offline training script."
        ) from exc

    model = YOLO(base_model)
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(project),
        name=name,
        val=True,
        plots=True,
        save=True,
    )

    save_dir = Path(getattr(results, "save_dir", project / name))
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise SystemExit(f"Training finished, but best weights were not found at {best_weights}")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Apricot YOLO yeast colony detector offline.")
    parser.add_argument("--data", type=Path, default=Path("data/generated/apricot_synthetic_suite_v1/dataset.yaml"))
    parser.add_argument(
        "--schema",
        default=DEFAULT_TRAINING_SCHEMA,
        help=(
            "Training specialist schema. Supported values: "
            + ", ".join(sorted(TRAINING_SPECIALISTS))
            + ". Defaults to clean_dots."
        ),
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default=None, help="Ultralytics run name. Defaults to the selected specialist.")
    parser.add_argument("--output", type=Path, default=None, help="Weights output path. Defaults to the selected specialist.")
    parser.add_argument(
        "--include-schema",
        action="append",
        default=[],
        help=(
            "Allow an additional dataset schema beyond the selected training schema. Repeat or pass "
            "comma-separated values only for intentional mixed-schema training."
        ),
    )
    parser.add_argument(
        "--allow-specialist-output-mismatch",
        action="store_true",
        help="Allow writing one specialist's weights to another specialist's protected output path.",
    )
    args = parser.parse_args()
    specialist = training_specialist_for_schema(args.schema)
    name = args.name or specialist.run_name
    output_path = args.output or specialist.output_path

    output = train_yolo(
        dataset_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=name,
        output=output_path,
        include_schemas=tuple(args.include_schema),
        training_schema=specialist.schema,
        allow_specialist_output_mismatch=args.allow_specialist_output_mismatch,
    )
    print(f"Copied best weights to {output}")


if __name__ == "__main__":
    main()
