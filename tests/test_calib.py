import unittest

import cv2
import numpy as np

from aoi.calib import calibrate, make_board, target_image

SIZE = (2736, 1824)
F_PX = 2500.0
WD = 127.0
D_TRUE = np.array([-0.06, 0.02, 0.0004, -0.0003, 0.0])


def _rot(ax, ay, az):
    return cv2.Rodrigues(np.radians([ax, ay, 0.0]))[0] @ cv2.Rodrigues(np.radians([0.0, 0.0, az]))[0]


def render_views(board, square_mm, poses, rng):
    ppm, margin_mm = 20.0, 5.0
    tgt = target_image(board, square_mm, dpi=int(ppm * 25.4), margin_mm=margin_mm)
    A = np.array([[1 / ppm, 0, -margin_mm], [0, 1 / ppm, -margin_mm], [0, 0, 1]])
    K = np.array([[F_PX, 0, SIZE[0] / 2 + 7], [0, F_PX, SIZE[1] / 2 - 5], [0, 0, 1]])
    sx, sy = board.getChessboardSize()
    centre = np.array([sx * square_mm / 2, sy * square_mm / 2, 0])
    xs, ys = np.meshgrid(np.arange(SIZE[0], dtype=np.float32), np.arange(SIZE[1], dtype=np.float32))
    ideal = cv2.undistortPoints(np.dstack([xs, ys]).reshape(-1, 1, 2), K, D_TRUE, P=K).reshape(SIZE[1], SIZE[0], 2)
    views = []
    for i, (ax, ay, az, ox, oy, dz) in enumerate(poses):
        R = _rot(ax, ay, az)
        t = -R @ centre + np.array([ox, oy, WD + dz])
        H = K @ np.c_[R[:, 0], R[:, 1], t] @ A
        img = cv2.warpPerspective(tgt, H, SIZE, flags=cv2.INTER_AREA, borderValue=140).astype(np.float32)
        img = cv2.remap(img, ideal[..., 0], ideal[..., 1], cv2.INTER_LINEAR, borderValue=140)
        img = cv2.GaussianBlur(img, (0, 0), 0.8) * 0.75 + 30 + rng.normal(0, 2, img.shape)
        views.append((f"v{i:02d}", np.clip(img, 0, 255).astype(np.uint8)))
    return views, K


class ChArUcoCalibration(unittest.TestCase):
    def test_recovers_intrinsics_wd_scale(self):
        board = make_board((12, 8), 8.0, 6.0)
        rng = np.random.default_rng(0)
        poses = [(0, 0, 0, 0, 0, 0)] + [(rng.uniform(-12, 12), rng.uniform(-12, 12), rng.uniform(-20, 20),
                                         rng.uniform(-20, 20), rng.uniform(-12, 12), rng.uniform(-3, 3))
                                        for _ in range(11)]
        views, K = render_views(board, 8.0, poses, rng)
        res = calibrate(views, board, log=lambda *_: None)
        self.assertLess(res["rms_px"], 0.3)
        self.assertAlmostEqual(res["K"][0, 0] / F_PX, 1.0, delta=0.005)
        self.assertAlmostEqual(res["K"][0, 2], K[0, 2], delta=3)
        self.assertAlmostEqual(res["D"].ravel()[0], D_TRUE[0], delta=0.01)
        self.assertEqual(res["flat_view"], "v00")
        self.assertAlmostEqual(res["wd_mm"], WD, delta=0.7)
        self.assertAlmostEqual(res["raw_px_per_mm"] / (F_PX / WD), 1.0, delta=0.005)


if __name__ == "__main__":
    unittest.main()
