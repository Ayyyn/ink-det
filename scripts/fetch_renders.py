#!/usr/bin/env python3
"""Fetch published surface volumes (renders) for an eligible scroll and write
inspectable PNGs — the zero-cost first look the docs ask for before any model.

The challenge's own guidance is that ink is sometimes visible directly in a
flattened render with no model at all, usually as bright areas, and that alone
qualifies for First Letters. Published surface volumes are uint8 and uncompressed,
so this is a download-and-look job: no GPU, no volume streaming, no geometry.

Per layer it writes a percentile-stretched PNG, and across depth it writes mean /
max / median composites, because a stroke too faint in one layer often survives
averaging over the sheet's thickness.

    python scripts/fetch_renders.py --scroll PHerc0800
    python scripts/fetch_renders.py --scroll PHerc0800 --segment 20251028222030 --keep-npy
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
for _n in ("s3fs", "aiobotocore", "botocore", "zarr"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

BUCKET = "vesuvius-challenge-open-data"


def s3():
    import s3fs

    return s3fs.S3FileSystem(anon=True)


def find_segments(scroll: str) -> list[dict]:
    """Locate every published segment of a scroll and its render + mesh URIs."""
    fs = s3()
    base = f"{BUCKET}/{scroll}/segments"
    if not fs.exists(base):
        return []
    out = []
    for seg in sorted(fs.ls(base)):
        name = seg.split("/")[-1]
        rec = {"segment": name, "render": None, "mesh": None, "volume": None}
        for sub, key in (("surface-volumes", "render"), ("mesh", "mesh")):
            p = f"{seg}/{sub}"
            if not fs.exists(p):
                continue
            for q in fs.ls(p):
                qn = q.split("/")[-1]
                if key == "render" and qn.endswith(".zarr"):
                    rec["render"] = "s3://" + q
                    # 8.64um-...-volume-<volume id>.zarr
                    rec["volume"] = qn.rsplit("-", 1)[-1].replace(".zarr", "")
                elif key == "mesh" and qn.endswith(".tifxyz"):
                    rec["mesh"] = "s3://" + q
        if rec["render"]:
            out.append(rec)
    return out


def stretch(a: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    """Percentile-stretch to uint8, ignoring the zero no-data background."""
    m = a > 0
    if not m.any():
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(a[m].astype(np.float32), [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255.0
    return (out * m).astype(np.uint8)


def write_png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path, optimize=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--segment", action="append", default=None,
                    help="substring of a segment name; repeatable. default: all")
    ap.add_argument("--out", default="renders", help="output root (gitignored)")
    ap.add_argument("--min-cm2", type=float, default=0.0,
                    help="skip segments whose canvas is smaller than this")
    ap.add_argument("--px-um", type=float, default=8.64)
    ap.add_argument("--keep-npy", action="store_true", help="also keep the raw uint8 stack")
    ap.add_argument("--layer-stride", type=int, default=1)
    args = ap.parse_args(argv)

    import zarr

    segs = find_segments(args.scroll)
    if not segs:
        print(f"no published segments for {args.scroll}", file=sys.stderr)
        return 1
    if args.segment:
        segs = [s for s in segs if any(f in s["segment"] for f in args.segment)]

    root = Path(args.out) / args.scroll
    root.mkdir(parents=True, exist_ok=True)
    manifest = []

    for rec in segs:
        g = zarr.open(rec["render"], mode="r", storage_options={"anon": True})
        arr = g["0"]
        nz, ny, nx = arr.shape
        cm2 = (ny * args.px_um / 1e4) * (nx * args.px_um / 1e4)
        tag = rec["segment"].split("-")[0]
        print(f"== {rec['segment']}  {nz}x{ny}x{nx}  {cm2:.2f} cm2  vol {rec['volume']}", flush=True)
        if cm2 < args.min_cm2:
            print("   skipped (too small)", flush=True)
            continue

        d = root / tag
        d.mkdir(exist_ok=True)
        t0 = time.time()
        stack = arr[:]  # uint8, uncompressed; the whole render is ~200 MB
        dt = time.time() - t0
        filled = float((stack > 0).mean())
        print(f"   downloaded {stack.nbytes/1e6:.0f} MB in {dt:.0f}s, {filled*100:.1f}% non-zero", flush=True)

        for z in range(0, nz, args.layer_stride):
            write_png(d / f"layer{z:02d}.png", stretch(stack[z]))
        valid = stack.astype(np.float32)
        valid[stack == 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_png(d / "composite_mean.png", stretch(np.nan_to_num(np.nanmean(valid, 0))))
            write_png(d / "composite_max.png", stretch(np.nan_to_num(np.nanmax(valid, 0))))
            write_png(d / "composite_median.png", stretch(np.nan_to_num(np.nanmedian(valid, 0))))
        if args.keep_npy:
            np.save(d / "stack_uint8.npy", stack)

        rec.update(shape=[int(nz), int(ny), int(nx)], cm2=round(cm2, 2),
                   nonzero_frac=round(filled, 4), out=str(d))
        manifest.append(rec)
        del stack, valid

    (root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\nwrote {len(manifest)} segment(s) under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
