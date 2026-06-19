from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gold_experience.annotations import (  # noqa: E402
    ColonyAnnotation,
    DishAnnotation,
    PlateGoldAnnotation,
    PolygonAnnotation,
)


SUPPORTED_LABELS = {
    "dish": "ellipse",
    "colony": "points",
    "colony_point": "points",
    "colony_ellipse": "ellipse",
    "label_region": "polygon",
    "ignore_region": "polygon",
}


def _normalize_label(raw: str) -> str:
    return raw.strip().lower()


def _is_colony_point_label(label: str) -> bool:
    return label in {"colony", "colony_point"}


def _is_colony_ellipse_label(label: str) -> bool:
    return label in {"colony_ellipse", "colony_blob"} or label == "colony"


def _parse_points_blob(raw: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_pos, y_pos = pair.split(",", maxsplit=1)
        points.append((float(x_pos), float(y_pos)))
    return points


def import_cvat_native(xml_path: Path) -> list[PlateGoldAnnotation]:
    tree = _parse_cvat_xml(xml_path)
    root = tree.getroot()
    records: list[PlateGoldAnnotation] = []

    for image_node in root.findall(".//image"):
        image_name = image_node.attrib["name"]
        width = int(float(image_node.attrib["width"]))
        height = int(float(image_node.attrib["height"]))
        record = PlateGoldAnnotation(
            image=image_name,
            image_width=width,
            image_height=height,
            metadata={
                "source": "cvat_native",
                "cvat_image_id": image_node.attrib.get("id"),
            },
        )

        for ellipse in image_node.findall("ellipse"):
            label = _normalize_label(ellipse.attrib.get("label", ""))
            if label == "dish":
                record.dish = DishAnnotation(
                    cx=float(ellipse.attrib["cx"]),
                    cy=float(ellipse.attrib["cy"]),
                    rx=float(ellipse.attrib["rx"]),
                    ry=float(ellipse.attrib["ry"]),
                    rotation=float(ellipse.attrib.get("rotation", 0.0)),
                )
                continue
            if not _is_colony_ellipse_label(label):
                continue
            rx = float(ellipse.attrib["rx"])
            ry = float(ellipse.attrib["ry"])
            record.colonies.append(
                ColonyAnnotation(
                    x=float(ellipse.attrib["cx"]),
                    y=float(ellipse.attrib["cy"]),
                    radius=(rx + ry) / 2.0,
                    morphology="ellipse",
                    rx=rx,
                    ry=ry,
                    rotation=float(ellipse.attrib.get("rotation", 0.0)),
                    tag=ellipse.attrib.get("occluded"),
                )
            )

        for points_node in image_node.findall("points"):
            label = _normalize_label(points_node.attrib.get("label", ""))
            if not _is_colony_point_label(label):
                continue
            points = _parse_points_blob(points_node.attrib["points"])
            if not points:
                continue
            x_pos, y_pos = points[0]
            record.colonies.append(
                ColonyAnnotation(
                    x=x_pos,
                    y=y_pos,
                    morphology="point",
                    tag=points_node.attrib.get("occluded"),
                )
            )

        for polygon_node in image_node.findall("polygon"):
            label = _normalize_label(polygon_node.attrib.get("label", ""))
            points = _parse_points_blob(polygon_node.attrib["points"])
            polygon = PolygonAnnotation(points=points)
            if label == "label_region":
                record.label_regions.append(polygon)
            elif label == "ignore_region":
                record.ignore_regions.append(polygon)

        records.append(record)

    return records


def _parse_cvat_xml(input_path: Path) -> ET.ElementTree:
    if input_path.suffix.lower() != ".zip":
        return ET.parse(input_path)

    with zipfile.ZipFile(input_path) as archive:
        annotation_members = [
            name
            for name in archive.namelist()
            if Path(name).name == "annotations.xml" and not name.endswith("/")
        ]
        if not annotation_members:
            raise ValueError(f"No annotations.xml found inside {input_path}")
        if len(annotation_members) > 1:
            annotation_members.sort(key=lambda item: (item.count("/"), item))
        with archive.open(annotation_members[0]) as handle:
            return ET.parse(handle)


def _record_summary(record: PlateGoldAnnotation) -> dict[str, int | str]:
    return {
        "image": record.image,
        "colonies": len(record.colonies),
        "point_colonies": sum(1 for item in record.colonies if item.morphology == "point"),
        "ellipse_colonies": sum(1 for item in record.colonies if item.morphology == "ellipse"),
        "label_regions": len(record.label_regions),
        "ignore_regions": len(record.ignore_regions),
        "has_dish": 1 if record.dish is not None else 0,
    }


def _print_summary(
    records: list[PlateGoldAnnotation],
    output_dir: Path,
    image_dir: Path | None,
) -> tuple[list[Path], list[Path]]:
    output_paths = [output_dir / f"{Path(record.image).stem}.gold.json" for record in records]
    existing_outputs = [path for path in output_paths if path.exists()]
    missing_images: list[Path] = []
    if image_dir is not None:
        missing_images = [image_dir / record.image for record in records if not (image_dir / record.image).exists()]

    total_colonies = sum(len(record.colonies) for record in records)
    total_points = sum(
        1 for record in records for item in record.colonies if item.morphology == "point"
    )
    total_ellipses = sum(
        1 for record in records for item in record.colonies if item.morphology == "ellipse"
    )
    total_label_regions = sum(len(record.label_regions) for record in records)
    total_ignore_regions = sum(len(record.ignore_regions) for record in records)
    missing_dish = [record.image for record in records if record.dish is None]

    print("CVAT import summary")
    print(f"  Images: {len(records)}")
    print(f"  Colonies: {total_colonies} ({total_points} point, {total_ellipses} ellipse)")
    print(f"  Label regions: {total_label_regions}")
    print(f"  Ignore regions: {total_ignore_regions}")
    print(f"  Missing dish ellipses: {len(missing_dish)}")
    print(f"  Existing gold files that would be overwritten: {len(existing_outputs)}")
    if image_dir is not None:
        print(f"  Missing benchmark images in {image_dir}: {len(missing_images)}")

    for record in records:
        summary = _record_summary(record)
        print(
            "  - "
            f"{summary['image']}: "
            f"{summary['colonies']} colonies "
            f"({summary['point_colonies']} point, {summary['ellipse_colonies']} ellipse), "
            f"{summary['label_regions']} label regions, "
            f"{summary['ignore_regions']} ignore regions"
        )

    if existing_outputs:
        print("\nExisting outputs:")
        for path in existing_outputs:
            print(f"  {path}")
    if missing_images:
        print("\nMissing images:")
        for path in missing_images:
            print(f"  {path}")
    if missing_dish:
        print("\nImages without dish ellipse:")
        for image_name in missing_dish:
            print(f"  {image_name}")

    return existing_outputs, missing_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Import CVAT native image annotations into Apricot gold annotation JSON.")
    parser.add_argument("xml_path", type=Path, help="Path to CVAT native annotations.xml or exported .zip")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations/gold"),
        help="Directory for generated *.gold.json files",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional benchmark image directory used to check that every annotated image exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and summarize the export without writing *.gold.json files.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if this import would replace an existing *.gold.json file.",
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Fail if any annotated image is missing from --image-dir.",
    )
    parser.add_argument(
        "--batch-name",
        default=None,
        help="Optional batch/source name to store in each gold annotation metadata record.",
    )
    args = parser.parse_args()

    records = import_cvat_native(args.xml_path)
    existing_outputs, missing_images = _print_summary(
        records=records,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
    )

    if args.no_overwrite and existing_outputs:
        raise SystemExit("Refusing to overwrite existing gold files. Re-run without --no-overwrite if intentional.")
    if args.require_images and args.image_dir is None:
        raise SystemExit("--require-images needs --image-dir.")
    if args.require_images and missing_images:
        raise SystemExit("Refusing import because one or more annotated images are missing from --image-dir.")
    if args.dry_run:
        print("Dry run complete. No files written.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        if args.batch_name:
            record.metadata["batch_name"] = args.batch_name
        output_path = args.output_dir / f"{Path(record.image).stem}.gold.json"
        record.save(output_path)
        print(f"Wrote {output_path}")

    print(
        "Imported "
        f"{len(records)} images from {args.xml_path.name}. "
        "Expected labels: dish, colony_point, colony_ellipse, label_region, ignore_region."
    )


if __name__ == "__main__":
    main()
