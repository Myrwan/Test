#!/usr/bin/env python3
"""Reposition a mouth sticker so it is centered on a person's mouth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _norm_to_px(value: float, size: int, scale: float) -> int:
    return int(round(value / scale * size))


def _parse_bbox(values: list[float]) -> tuple[float, float, float, float]:
    ymin, xmin, ymax, xmax = values
    return ymin, xmin, ymax, xmax


def _bbox_to_pixels(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    scale: float,
) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = bbox
    y1 = _norm_to_px(ymin, height, scale)
    x1 = _norm_to_px(xmin, width, scale)
    y2 = _norm_to_px(ymax, height, scale)
    x2 = _norm_to_px(xmax, width, scale)
    return x1, y1, x2, y2


def _center(bbox_px: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox_px
    return (x1 + x2) // 2, (y1 + y2) // 2


def reposition_sticker(
    image: np.ndarray,
    source_bbox: tuple[float, float, float, float],
    target_center: tuple[float, float],
    coord_scale: float = 1000.0,
    inpaint_radius: int = 5,
) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _bbox_to_pixels(source_bbox, width, height, coord_scale)

    sticker = image[y1:y2, x1:x2].copy()
    if sticker.size == 0:
        raise ValueError("Sticker bounding box is empty; check the coordinates.")

    sticker_h, sticker_w = sticker.shape[:2]
    src_cx, src_cy = _center((x1, y1, x2, y2))
    tgt_cx = _norm_to_px(target_center[0], width, coord_scale)
    tgt_cy = _norm_to_px(target_center[1], height, coord_scale)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    restored = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)

    new_x1 = tgt_cx - sticker_w // 2
    new_y1 = tgt_cy - sticker_h // 2
    new_x2 = new_x1 + sticker_w
    new_y2 = new_y1 + sticker_h

    result = restored.copy()
    dst_x1 = max(0, new_x1)
    dst_y1 = max(0, new_y1)
    dst_x2 = min(width, new_x2)
    dst_y2 = min(height, new_y2)

    src_x1 = dst_x1 - new_x1
    src_y1 = dst_y1 - new_y1
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    roi = result[dst_y1:dst_y2, dst_x1:dst_x2]
    patch = sticker[src_y1:src_y2, src_x1:src_x2]

    if patch.size == 0:
        raise ValueError("Target position places the sticker outside the image.")

    center = (patch.shape[1] // 2, patch.shape[0] // 2)
    result[dst_y1:dst_y2, dst_x1:dst_x2] = cv2.seamlessClone(
        patch,
        roi,
        np.full(patch.shape[:2], 255, dtype=np.uint8),
        center,
        cv2.NORMAL_CLONE,
    )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Center a mouth sticker on a person's mouth in a photo."
    )
    parser.add_argument("input_image", type=Path, help="Source photo")
    parser.add_argument("output_image", type=Path, help="Edited photo")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("YMIN", "XMIN", "YMAX", "XMAX"),
        default=[510, 220, 710, 510],
        help="Current sticker bounding box on a 0-1000 scale",
    )
    parser.add_argument(
        "--mouth",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=[360, 580],
        help="Target mouth center on a 0-1000 scale",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1000.0,
        help="Coordinate scale used for bbox and mouth values",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.input_image))
    if image is None:
        print(f"Could not read image: {args.input_image}", file=sys.stderr)
        return 1

    output = reposition_sticker(
        image,
        _parse_bbox(args.bbox),
        (args.mouth[0], args.mouth[1]),
        coord_scale=args.scale,
    )

    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_image), output)
    print(f"Saved: {args.output_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
