"""Shared annotation parsing utilities for COCO and YOLO formats.

Provides a unified ``AnnotatedDataset`` structure that downstream
detectors can consume regardless of the original annotation format.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BBox:
    """Bounding box in normalised (YOLO-style) coordinates."""

    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class ObjectAnnotation:
    """A single annotated object in an image."""

    class_id: int
    class_name: str | None = None
    bbox: BBox | None = None


@dataclass
class AnnotatedImage:
    """An image together with its object annotations."""

    image_path: str
    annotations: list[ObjectAnnotation] = field(default_factory=list)

    @property
    def class_ids(self) -> list[int]:
        """All class IDs present in this image's annotations."""
        return [a.class_id for a in self.annotations]


@dataclass
class AnnotatedDataset:
    """A full dataset of annotated images."""

    images: list[AnnotatedImage] = field(default_factory=list)
    class_names: dict[int, str] = field(default_factory=dict)

    @property
    def image_paths(self) -> list[str]:
        return [img.image_path for img in self.images]

    @property
    def image_level_labels(self) -> list[int]:
        """Return the dominant (most-frequent) class ID per image.

        For images with no annotations the label is ``-1``.
        """
        labels: list[int] = []
        for img in self.images:
            if not img.annotations:
                labels.append(-1)
            else:
                c = Counter(a.class_id for a in img.annotations)
                labels.append(c.most_common(1)[0][0])
        return labels


# ── YOLO parser ──────────────────────────────────────────────────────────────

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def parse_yolo(
    image_dir: str | Path,
    label_dir: str | Path,
    class_names: dict[int, str] | None = None,
    image_extensions: tuple[str, ...] = _IMAGE_EXTS,
) -> AnnotatedDataset:
    """Parse a YOLO-format annotation directory.

    Each image in *image_dir* should have a corresponding ``.txt`` file in
    *label_dir* sharing the same stem.  Every line in the label file:

        ``class_id  center_x  center_y  width  height``

    All coordinate values are normalised to ``[0, 1]``.
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    class_names = class_names or {}

    images: list[AnnotatedImage] = []

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue

        label_path = label_dir / (img_path.stem + ".txt")
        annotations: list[ObjectAnnotation] = []

        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                annotations.append(
                    ObjectAnnotation(
                        class_id=cid,
                        class_name=class_names.get(cid),
                        bbox=BBox(
                            x_center=float(parts[1]),
                            y_center=float(parts[2]),
                            width=float(parts[3]),
                            height=float(parts[4]),
                        ),
                    )
                )

        images.append(AnnotatedImage(image_path=str(img_path), annotations=annotations))

    return AnnotatedDataset(images=images, class_names=class_names)


# ── COCO parser ──────────────────────────────────────────────────────────────

def parse_coco(
    annotation_file: str | Path,
    image_dir: str | Path,
) -> AnnotatedDataset:
    """Parse a COCO-format annotation JSON file.

    Uses ``pycocotools`` for robust handling of the standard COCO schema.
    Bounding boxes are converted from COCO pixel format
    ``[x_min, y_min, w, h]`` to normalised YOLO-style coordinates.
    """
    from pycocotools.coco import COCO  # type: ignore[import-untyped]

    annotation_file = Path(annotation_file)
    image_dir = Path(image_dir)

    coco = COCO(str(annotation_file))

    class_names: dict[int, str] = {
        cat["id"]: cat["name"] for cat in coco.cats.values()
    }

    images: list[AnnotatedImage] = []

    for img_id in sorted(coco.imgs.keys()):
        img_info = coco.imgs[img_id]
        img_path = image_dir / img_info["file_name"]
        img_w = img_info.get("width", 1)
        img_h = img_info.get("height", 1)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_anns = coco.loadAnns(ann_ids)

        annotations: list[ObjectAnnotation] = []
        for ann in coco_anns:
            cid = ann["category_id"]
            bbox = None
            if "bbox" in ann:
                x_min, y_min, bw, bh = ann["bbox"]
                bbox = BBox(
                    x_center=(x_min + bw / 2) / img_w,
                    y_center=(y_min + bh / 2) / img_h,
                    width=bw / img_w,
                    height=bh / img_h,
                )
            annotations.append(
                ObjectAnnotation(
                    class_id=cid,
                    class_name=class_names.get(cid),
                    bbox=bbox,
                )
            )

        images.append(AnnotatedImage(image_path=str(img_path), annotations=annotations))

    return AnnotatedDataset(images=images, class_names=class_names)
