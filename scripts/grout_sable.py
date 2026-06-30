#!/usr/bin/env python3
"""Recolor dark tile joints to sand-colored grout."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


SAND_BGR = (160, 181, 196)  # warm beige in BGR (~#C4B5A0)


def recolor_joints(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)

    # Dark recessed joints and spacers.
    dark_mask = (l_channel < 72).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    dark_mask = cv2.dilate(dark_mask, kernel, iterations=1)

    # Keep only thin joint-like structures.
    thin = cv2.ximgproc.thinning(dark_mask) if hasattr(cv2, "ximgproc") else dark_mask
    joint_mask = thin if thin is not None and thin.any() else dark_mask

    result = image.copy()
    sand = np.full_like(image, SAND_BGR, dtype=np.uint8)

    # Soft blend at joint edges for a natural grout look.
    blurred_mask = cv2.GaussianBlur(joint_mask.astype(np.float32), (5, 5), 0)
    alpha = np.clip(blurred_mask / 255.0, 0, 1)[..., None]
    result = (result.astype(np.float32) * (1 - alpha) + sand.astype(np.float32) * alpha).astype(
        np.uint8
    )

    return result


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: grout_sable.py <input_image> <output_image>")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Could not read image: {input_path}")
        return 1

    output = recolor_joints(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
