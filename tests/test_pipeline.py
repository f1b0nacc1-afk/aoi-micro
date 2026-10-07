import shutil
import tempfile
import unittest
from pathlib import Path

from aoi.__main__ import _components, _inspect_file
from aoi.camera import list_images, read_gray
from aoi.config import load_config
from aoi.model import train
from aoi.synth import make_demo

EXPECTED = {"R3": "MISMATCH", "C2": "SHIFT", "D1": "POLARITY", "Q1": "ROTATED",
            "U1": "POLARITY", "U2": "ANOMALY"}


class SyntheticBoard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        make_demo(cls.tmp, n_golden=12)
        cls.cfg = load_config(cls.tmp / "config.json")
        imgs = [read_gray(f) for f in list_images(str(cls.tmp / "golden"))]
        train(imgs, _components(cls.cfg), cls.cfg, cls.tmp / "model", log=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def _run(self, name):
        import json
        _inspect_file(self.cfg, self.tmp / "test" / name, self.tmp / "model", self.tmp / "out" / name)
        return json.loads((self.tmp / "out" / name / "result.json").read_text())

    def test_good_boards_pass(self):
        for i in range(5):
            r = self._run(f"good_{i}.png")
            self.assertEqual(r["summary"]["verdict"], "PASS", r["summary"])

    def test_bad_board_defects(self):
        r = self._run("bad.png")
        fails = {p["ref"]: p["defects"] for p in r["parts"] if p["status"] == "FAIL"}
        self.assertEqual(set(fails), set(EXPECTED))
        for ref, d in EXPECTED.items():
            self.assertIn(d, fails[ref], ref)
        spots = {(round(a["x_mm"]), round(a["y_mm"])) for a in r["board_anomalies"]}
        self.assertIn((24, 41), spots)
        self.assertIn((10, 20), spots)


if __name__ == "__main__":
    unittest.main()
