"""Inference: full-image paragraph-corner detection + reconstruction.

Pipeline:
    1. Slide an ``n x n`` window over the image (same geometry as extraction).
    2. Each patch yields, per corner type, an existence probability and an
       (x, y) position; keep detections above ``threshold`` and map them to
       global image coordinates.
    3. Merge duplicate detections of the same corner type across overlapping
       patches (distance-based NMS).
    4. Reconstruct axis-aligned paragraph boxes from the typed corners
       (greedy TL->BR pairing). This is a heuristic that improves with a
       better-trained model; the raw typed corners are always returned too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from itf.datasets.loader import CORNER_NAMES
from itf.models.builder import build_model
from itf.patches.extract import _positions


@dataclass
class Detection:
    corner: str          # TL | TR | BR | BL
    x: float             # global px
    y: float             # global px
    score: float

    def as_dict(self) -> dict:
        return {"corner": self.corner, "x": round(self.x, 2), "y": round(self.y, 2),
                "score": round(self.score, 4)}


def load_model(checkpoint_path: str | Path, device: str = "cpu"):
    ckpt = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model_cfg = ckpt["config"]["model"]
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def detect_corners(model, image: np.ndarray, patch_size: int | None = None,
                   stride: int | None = None, threshold: float = 0.5,
                   device: str = "cpu") -> list[Detection]:
    """Run the sliding window and return merged corner detections in image px."""
    if image.ndim == 3:
        image = image[..., 0]
    n = patch_size or model.config.input_size
    stride = stride or max(n // 2, 1)
    h, w = image.shape

    ys = _positions(h, n, stride)
    xs = _positions(w, n, stride)

    patches, offsets, borders = [], [], []
    for y0 in ys:
        for x0 in xs:
            patches.append(image[y0:y0 + n, x0:x0 + n])
            offsets.append((x0, y0))
            # Same geometry/order as extraction (BORDER_NAMES): top,right,bottom,left.
            borders.append((int(y0 == 0), int(x0 + n >= w),
                            int(y0 + n >= h), int(x0 == 0)))
    batch = torch.from_numpy(np.stack(patches)).unsqueeze(1).float().div_(255.0).to(device)
    border_batch = torch.tensor(borders, dtype=torch.float32, device=device)  # (P, 4)
    preds = model(batch, border_batch)         # (P, 4, 3)
    probs = torch.sigmoid(preds[..., 0])       # (P, 4)

    raw: list[Detection] = []
    for i, (x0, y0) in enumerate(offsets):
        for c in range(len(CORNER_NAMES)):
            score = float(probs[i, c])
            if score >= threshold:
                gx = x0 + float(preds[i, c, 1]) * n
                gy = y0 + float(preds[i, c, 2]) * n
                raw.append(Detection(CORNER_NAMES[c], gx, gy, score))
    return _merge(raw, radius=stride / 2.0)


def _merge(dets: list[Detection], radius: float) -> list[Detection]:
    """Greedy distance-based NMS per corner type; averages merged positions."""
    merged: list[Detection] = []
    for corner in CORNER_NAMES:
        group = sorted([d for d in dets if d.corner == corner],
                       key=lambda d: d.score, reverse=True)
        used = [False] * len(group)
        for i, d in enumerate(group):
            if used[i]:
                continue
            cluster = [d]
            used[i] = True
            for j in range(i + 1, len(group)):
                if not used[j] and abs(group[j].x - d.x) <= radius and abs(group[j].y - d.y) <= radius:
                    cluster.append(group[j])
                    used[j] = True
            wsum = sum(c.score for c in cluster)
            mx = sum(c.x * c.score for c in cluster) / wsum
            my = sum(c.y * c.score for c in cluster) / wsum
            merged.append(Detection(corner, mx, my, max(c.score for c in cluster)))
    return merged


def reconstruct_boxes(dets: list[Detection], min_size: float = 4.0) -> list[dict]:
    """Greedy TL->BR pairing into axis-aligned boxes [x, y, w, h].

    For each top-left, pick the nearest bottom-right below-and-right of it;
    optionally snap width/height using TR/BL when available.
    """
    tls = [d for d in dets if d.corner == "TL"]
    brs = [d for d in dets if d.corner == "BR"]
    used_br = [False] * len(brs)
    boxes: list[dict] = []
    for tl in sorted(tls, key=lambda d: (d.y, d.x)):
        best_j, best_d = -1, float("inf")
        for j, br in enumerate(brs):
            if used_br[j]:
                continue
            if br.x > tl.x + min_size and br.y > tl.y + min_size:
                dist = (br.x - tl.x) + (br.y - tl.y)
                if dist < best_d:
                    best_d, best_j = dist, j
        if best_j >= 0:
            used_br[best_j] = True
            br = brs[best_j]
            boxes.append({
                "box": [round(tl.x, 2), round(tl.y, 2),
                        round(br.x - tl.x, 2), round(br.y - tl.y, 2)],
                "score": round((tl.score + br.score) / 2, 4),
            })
    return boxes


def predict_image(model, image: np.ndarray, patch_size: int | None = None,
                  stride: int | None = None, threshold: float = 0.5,
                  device: str = "cpu") -> dict:
    dets = detect_corners(model, image, patch_size, stride, threshold, device)
    return {
        "corners": [d.as_dict() for d in dets],
        "paragraphs": reconstruct_boxes(dets),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
    }
