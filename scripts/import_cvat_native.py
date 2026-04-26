from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gold_experience.annotations import (  # noqa: E402
    DishAnnotation,
    PlateGoldAnnotation,
    PointAnnotation,
    PolygonAnnotation,
)


SUPPORTED_LABELS = {
    "dish": "ellipse",
    "colony": "points",
    "label_region": "polygon",
    "ignore_region": "polygon",
}


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
    tree = ET.parse(xml_path)
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
            label = ellipse.attrib.get("label", "")
            if label != "dish":
                continue
            record.dish = DishAnnotation(
                cx=float(ellipse.attrib["cx"]),
                cy=float(ellipse.attrib["cy"]),
                rx=float(ellipse.attrib["rx"]),
                ry=float(ellipse.attrib["ry"]),
                rotation=float(ellipse.attrib.get("rotation", 0.0)),
            )

        for points_node in image_node.findall("points"):
            label = points_node.attrib.get("label", "")
            if label != "colony":
                continue
            points = _parse_points_blob(points_node.attrib["points"])
            if not points:
                continue
            x_pos, y_pos = points[0]
            record.colonies.append(
                PointAnnotation(
                    x=x_pos,
                    y=y_pos,
                    tag=points_node.attrib.get("occluded"),
                )
            )

        for polygon_node in image_node.findall("polygon"):
            label = polygon_node.attrib.get("label", "")
            points = _parse_points_blob(polygon_node.attrib["points"])
            polygon = PolygonAnnotation(points=points)
            if label == "label_region":
                record.label_regions.append(polygon)
            elif label == "ignore_region":
                record.ignore_regions.append(polygon)

        records.append(record)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Import CVAT native image XML into GxP gold annotation JSON.")
    parser.add_argument("xml_path", type=Path, help="Path to CVAT native annotations.xml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations/gold"),
        help="Directory for generated *.gold.json files",
    )
    args = parser.parse_args()

    records = import_cvat_native(args.xml_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        output_path = args.output_dir / f"{Path(record.image).stem}.gold.json"
        record.save(output_path)
        print(f"Wrote {output_path}")

    print(
        "Imported "
        f"{len(records)} images from {args.xml_path.name}. "
        "Expected labels: dish, colony, label_region, ignore_region."
    )


if __name__ == "__main__":
    main()
