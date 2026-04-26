# GxP Bio Annotation Schema

Date: 2026-04-26

## Overview

We intentionally separate:

- **gold annotations** from CVAT
- **user hints** from the live GxP upload flow

Gold annotations are trusted.
User hints are weak supervision.

## Gold Annotation File

Suggested path:

- `data/annotations/gold/<image_stem>.gold.json`

Example:

```json
{
  "image": "1399.jpg",
  "image_width": 3790,
  "image_height": 4000,
  "dish": {
    "cx": 1895.0,
    "cy": 2000.0,
    "rx": 1620.0,
    "ry": 1620.0,
    "rotation": 0.0
  },
  "colonies": [
    { "x": 1203.5, "y": 998.0 },
    { "x": 1244.0, "y": 1032.0 }
  ],
  "label_regions": [
    {
      "points": [[1800.0, 900.0], [2050.0, 900.0], [2050.0, 2600.0], [1800.0, 2600.0]]
    }
  ],
  "ignore_regions": [],
  "metadata": {
    "source": "cvat_native",
    "annotator": "initials"
  }
}
```

## User Hint File

Suggested path:

- `data/annotations/hints/<image_stem>.hints.json`

Example:

```json
{
  "image": "1399.jpg",
  "image_width": 3790,
  "image_height": 4000,
  "positive_clicks": [
    { "x": 1220.0, "y": 1001.0 },
    { "x": 1522.0, "y": 1728.0 }
  ],
  "negative_clicks": [
    { "x": 1905.0, "y": 1502.0, "tag": "label_dot" }
  ],
  "label_hints": [],
  "metadata": {
    "source": "gxp_user",
    "model_version": "gold-experience-v1",
    "lab_id": "byu-test",
    "created_at": "2026-04-26T12:00:00Z"
  }
}
```

## Why Separate Them

Gold annotations:

- used for benchmark evaluation
- used for threshold tuning
- used for future supervised learning

User hints:

- used for per-image adaptation
- used for active-learning queues
- should not automatically become gold labels

## Code Support

Python models live in:

- `src/gold_experience/annotations.py`

Current classes:

- `DishAnnotation`
- `PointAnnotation`
- `PolygonAnnotation`
- `PlateGoldAnnotation`
- `PlateUserHints`
