from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from .cad import load_bom, load_centroid, load_packages
from .config import load_config, resolve


def _components(cfg):
    f = cfg["files"]
    lib = load_packages(resolve(cfg, f["packages"]))
    return load_centroid(resolve(cfg, f["centroid"]), lib, load_bom(resolve(cfg, f["bom"])))


def _grab(cfg, out: Path) -> Path:
    from .camera import grab_averaged, open_camera
    cam = open_camera(cfg["camera"])
    try:
        img = grab_averaged(cam, int(cfg["camera"]["average_frames"]))
    finally:
        cam.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out


def cmd_optics(a):
    from .optics import optics_table
    print(optics_table(a.focal, 2.4, (5472, 3648), a.wd, [2.6, 4, 5.6, 8], a.height, a.pupil))


def cmd_synth(a):
    from .synth import make_demo
    make_demo(Path(a.out), a.n)
    print(f"demo written to {a.out}: golden/*.png, test/good.png, test/bad.png, config.json")


def cmd_bench(a):
    import json as _json
    from .bench.run import run_bench
    over = _json.loads(a.override) if a.override else {}
    d = run_bench(Path(a.out), a.tag, a.designs, a.golden, a.good, a.alt, a.bad, a.per_bad, a.seed, over)
    print(f"-> {d / 'index.html'}")


def cmd_capture(a):
    cfg = load_config(a.config)
    for i in range(a.n):
        if a.n > 1:
            input(f"[{i + 1}/{a.n}] положите плату и нажмите Enter...")
        p = _grab(cfg, Path(a.out) / f"{a.prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{i:02d}.png")
        print(f"saved {p}")


def cmd_fiducials(a):
    from .align import find_all_fiducials
    from .camera import read_gray
    cfg = load_config(a.config)
    img = read_gray(a.image)
    b = cfg["board"]
    found = sorted(find_all_fiducials(img, b["fiducial_diameter_mm"] * b["raw_px_per_mm"]),
                   key=lambda f: -f[2])
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, (x, y, circ) in enumerate(found):
        print(f"  #{i}: approx_px [{x:.0f}, {y:.0f}]  circularity {circ:.2f}")
        cv2.circle(vis, (int(x), int(y)), 40, (0, 0, 255), 4)
        cv2.putText(vis, f"#{i}", (int(x) + 45, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    out = Path(a.image).with_suffix(".fiducials.jpg")
    cv2.imwrite(str(out), vis)
    print(f"{len(found)} candidates, preview {out}. Впишите approx_px нужных в board.fiducials.")


def cmd_train(a):
    from .camera import Undistorter, list_images, read_gray
    from .model import train
    cfg = load_config(a.config)
    und = Undistorter(resolve(cfg, cfg["optics"]["undistort_npz"]))
    files = list_images(a.images)
    if len(files) < 5:
        sys.exit(f"need >= 5 golden images (20-30 recommended), got {len(files)}")
    print(f"training on {len(files)} golden boards")
    meta = train([und(read_gray(f)) for f in files], _components(cfg), cfg, Path(a.golden))
    print(f"model: {len(meta['parts'])} parts -> {a.golden}")


def _inspect_file(cfg, path: Path, golden: Path, out: Path):
    from .align import BoardFrame
    from .camera import Undistorter, read_gray
    from .inspect import inspect_board
    from .model import load_model
    from .report import write_report
    meta, arrays = load_model(golden)
    raw = Undistorter(resolve(cfg, cfg["optics"]["undistort_npz"]))(read_gray(path))
    t0 = time.time()
    board, info, results, extra = inspect_board(raw, _components(cfg), meta, arrays, cfg)
    dt = time.time() - t0
    summary = write_report(out, board, info, results, extra, BoardFrame(cfg), path.name)
    for r in results:
        if r["status"] != "PASS":
            print(f"  {r['ref']:6s} {r['status']:5s} {' '.join(r['defects'])}")
    for e in extra:
        print(f"  board  FAIL  ANOMALY at ({e['x_mm']}, {e['y_mm']}) mm, {e['area_mm2']} mm²")
    print(f"{summary['verdict']}: {summary['n_fail_parts']} parts, {summary['n_board_anomalies']} board "
          f"anomalies, {dt:.1f} s -> {out / 'report.html'}")
    return summary


def cmd_inspect(a):
    from .camera import list_images
    cfg = load_config(a.config)
    for f in list_images(a.images):
        _inspect_file(cfg, f, Path(a.golden), Path(a.out) / f.stem)


def cmd_run(a):
    cfg = load_config(a.config)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    p = _grab(cfg, Path(a.out) / stamp / "raw.png")
    _inspect_file(cfg, p, Path(a.golden), p.parent)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aoi")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("optics", help="поле, мкм/px, ГРИП и параллакс по WD")
    s.add_argument("--focal", type=float, default=12.0)
    s.add_argument("--wd", type=float, nargs="+", default=[100, 150, 200, 300])
    s.add_argument("--height", type=float, default=2.0, help="высота детали для параллакса, мм")
    s.add_argument("--pupil", type=float, default=27.0,
                   help="передняя линза -> главная плоскость, мм (27 для V1226-MPZ)")
    s.set_defaults(fn=cmd_optics)

    s = sub.add_parser("synth", help="синтетическая демо-плата")
    s.add_argument("--out", default="demo")
    s.add_argument("--n", type=int, default=20)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("bench", help="реалистичный синтетический стенд с метриками")
    s.add_argument("--tag", default="run")
    s.add_argument("--out", default="bench/runs")
    s.add_argument("--designs", type=int, default=4)
    s.add_argument("--golden", type=int, default=20)
    s.add_argument("--good", type=int, default=2)
    s.add_argument("--alt", type=int, default=2, help="хорошие платы другой партии компонентов")
    s.add_argument("--bad", type=int, default=4)
    s.add_argument("--per-bad", type=int, default=6)
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--override", help='JSON для секции inspect, напр. \'{"var_k": 6}\'')
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("capture", help="снять кадр(ы) с камеры")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--out", default="captures")
    s.add_argument("--prefix", default="board")
    s.add_argument("--n", type=int, default=1)
    s.set_defaults(fn=cmd_capture)

    s = sub.add_parser("fiducials", help="найти кандидаты фидуциалов на кадре (первичная настройка)")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("image")
    s.set_defaults(fn=cmd_fiducials)

    s = sub.add_parser("train", help="обучить эталон по golden-платам")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--images", required=True, help="папка или glob")
    s.add_argument("--golden", default="golden_model")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("inspect", help="проверить снимки")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--images", required=True)
    s.add_argument("--golden", default="golden_model")
    s.add_argument("--out", default="reports")
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("run", help="снять и проверить")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--golden", default="golden_model")
    s.add_argument("--out", default="reports")
    s.set_defaults(fn=cmd_run)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
