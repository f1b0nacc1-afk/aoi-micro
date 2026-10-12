"""Tiled inference of the component detector on full-resolution mono frames (20 MP)."""
from __future__ import annotations

import cv2
import numpy as np


def _starts(n: int, tile: int, stride: int) -> list:
    if n <= tile:
        return [0]
    s = list(range(0, n - tile, stride))
    return s + [n - tile]


def detect_tiled(model, img: np.ndarray, tile: int = 1024, overlap: int = 512, conf: float = 0.25,
                 iou: float = 0.5, imgsz: int = 1024, batch: int = 8) -> np.ndarray:
    """Rows (x0, y0, x1, y1, conf, cls) in image px after class-wise NMS across tiles.
    Boxes touching an inner tile edge are dropped, so parts larger than overlap px can be lost."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    stride = tile - overlap
    origins = [(x, y) for y in _starts(h, tile, stride) for x in _starts(w, tile, stride)]
    rows = []
    for i in range(0, len(origins), batch):
        chunk = origins[i:i + batch]
        res = model.predict([img[y:y + tile, x:x + tile] for x, y in chunk], imgsz=imgsz, conf=conf,
                            iou=iou, verbose=False)
        for (x, y), r in zip(chunk, res):
            b = r.boxes
            if b is None or not len(b):
                continue
            xyxy = b.xyxy.cpu().numpy()
            th, tw = r.orig_shape
            edge = 4
            keep = ~(((xyxy[:, 0] < edge) & (x > 0)) | ((xyxy[:, 1] < edge) & (y > 0)) |
                     ((xyxy[:, 2] > tw - edge) & (x + tw < w)) | ((xyxy[:, 3] > th - edge) & (y + th < h)))
            xyxy = xyxy[keep] + [x, y, x, y]
            rows.append(np.c_[xyxy, b.conf.cpu().numpy()[keep], b.cls.cpu().numpy()[keep]])
    if not rows:
        return np.zeros((0, 6), np.float32)
    d = np.vstack(rows).astype(np.float32)
    out = []
    for c in np.unique(d[:, 5]):
        dc = d[d[:, 5] == c]
        boxes = [[float(a), float(b), float(cc - a), float(dd - b)] for a, b, cc, dd in dc[:, :4]]
        idx = cv2.dnn.NMSBoxes(boxes, dc[:, 4].tolist(), conf, iou)
        out.append(dc[np.array(idx).ravel()] if len(idx) else dc[:0])
    return np.vstack(out)
