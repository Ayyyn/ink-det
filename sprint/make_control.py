#!/usr/bin/env python3
"""Build the positive control: a cropped PHerc0139 w035 render plus its ink labels.

Every null result from this pipeline is uninterpretable without a known positive run
through the identical path. w035 is the right one: the challenge's own ink tutorial
uses it, its 28-layer 9.362 um surface volume is published, and so are hand-drawn ink
labels and the supervision mask marking where those labels are valid.

Two honest caveats about what this control proves:

  * All four models have seen PHerc0139 in training, and dense_native is a fine-tune on
    it. So this measures "does the pipeline recover ink that is there", NOT "does it
    generalise". It calibrates appearance and sanity, not expected performance on an
    unread scroll.
  * The published render is 26.7 cm2. Sweeping it whole costs ~5x a PHerc0800 segment,
    so it is cropped to a window over the labelled region, which also makes the two
    directly comparable in size.

    python sprint/make_control.py --out /data/control
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
for _n in ("s3fs", "aiobotocore", "botocore", "zarr"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

SEG = ("s3://vesuvius-challenge-open-data/PHerc0139/segments/"
       "20260317000000-w035_2026031718")
RENDER = f"{SEG}/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr"
LABELS = f"{SEG}/ink-labels/9.362um-volume-20250728140407/20260918"
PX_UM = 9.362


def write_zarr_v2_uint8(path: Path, block: np.ndarray, chunk: int = 128) -> None:
    """Write a zarr v2 store by hand: uint8, uncompressed, one chunk per (y, x) tile.

    Deliberately not using the zarr API. zarr 2.x and 3.x disagree on how a group is
    created (`create_dataset` vs `create_array`) and on which metadata file a store
    gets (`.zarray` vs `zarr.json`), and the GPU box does not necessarily have the
    version this was written against. Erwin's run_ink.sh keys off "0/.zarray", so the
    format is not negotiable; writing ~15 lines of JSON and raw bytes removes the
    dependency. This is byte-for-byte the layout the published renders use.
    """
    nz, ny, nx = (int(v) for v in block.shape)
    shutil.rmtree(path, ignore_errors=True)   # never inherit a half-written or v3 store
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}), encoding="utf-8")
    arr_dir = path / "0"
    arr_dir.mkdir(exist_ok=True)
    (arr_dir / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "shape": [nz, ny, nx], "chunks": [nz, chunk, chunk],
        "dtype": "|u1", "compressor": None, "fill_value": 0, "order": "C",
        "filters": None, "dimension_separator": "/",
    }, indent=1), encoding="utf-8")
    for iy in range((ny + chunk - 1) // chunk):
        d = arr_dir / "0" / str(iy)
        d.mkdir(parents=True, exist_ok=True)
        for ix in range((nx + chunk - 1) // chunk):
            buf = np.zeros((nz, chunk, chunk), np.uint8)   # v2 chunks are fixed size, zero padded
            sub = block[:, iy * chunk:(iy + 1) * chunk, ix * chunk:(ix + 1) * chunk]
            buf[:, :sub.shape[1], :sub.shape[2]] = sub
            (d / str(ix)).write_bytes(buf.tobytes(order="C"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=2500, help="square crop side in px (default 2500)")
    ap.add_argument("--name", default="w035_control")
    a = ap.parse_args(argv)

    import zarr

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    so = {"anon": True}

    def open2d(uri):
        """Label stores are published as a group holding array "0"; accept a bare array too."""
        o = zarr.open(uri, mode="r", storage_options=so)
        if hasattr(o, "keys") and "0" in o:
            o = o["0"]
        return np.asarray(o[:])

    lab = open2d(f"{LABELS}/inklabels.zarr")
    sup = open2d(f"{LABELS}/supervision.zarr")
    ys, xs = np.where(sup > 0)
    if not len(ys):
        print("supervision mask is empty", file=sys.stderr)
        return 1

    # centre the crop on the supervised region, clipped to the canvas
    H, W = lab.shape
    cy, cx = int((ys.min() + ys.max()) // 2), int((xs.min() + xs.max()) // 2)
    h = min(a.size, H, W)
    y0 = max(0, min(cy - h // 2, H - h))
    x0 = max(0, min(cx - h // 2, W - h))
    y1, x1 = y0 + h, x0 + h
    print(f"canvas {H}x{W}; supervised y {ys.min()}-{ys.max()} x {xs.min()}-{xs.max()}")
    print(f"crop   y {y0}-{y1}  x {x0}-{x1}  ({h} px = {h * PX_UM / 1e4:.2f} cm per side)")

    src = zarr.open(RENDER, mode="r", storage_options=so)["0"]
    nz = src.shape[0]
    print(f"pulling {nz} layers of the crop ...", flush=True)
    block = np.asarray(src[:, y0:y1, x0:x1])

    zpath = out / f"{a.name}.zarr"
    write_zarr_v2_uint8(zpath, block)
    # run_ink.sh and render_info.py read the pixel size from here
    (zpath / ".zattrs").write_text(json.dumps({
        "canvas_size": [int(h), int(h)],
        "multiscales": [{"axes": [{"name": n, "type": "space", "unit": "micrometer"} for n in "zyx"],
                         "datasets": [{"path": "0", "coordinateTransformations": [
                             {"type": "scale", "scale": [PX_UM, PX_UM, PX_UM]}]}]}],
    }, indent=1), encoding="utf-8")

    np.save(out / f"{a.name}__inklabels.npy", lab[y0:y1, x0:x1])
    np.save(out / f"{a.name}__supervision.npy", sup[y0:y1, x0:x1])
    (out / f"{a.name}__crop.json").write_text(json.dumps(
        {"segment": SEG, "render": RENDER, "labels": LABELS,
         "y0": int(y0), "y1": int(y1), "x0": int(x0), "x1": int(x1),
         "px_um": PX_UM, "n_layers": int(nz)}, indent=1), encoding="utf-8")

    # read it back the way the sweep will, and fail here rather than in run_ink.sh
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_info import info as render_info
    chk = render_info(str(zpath))
    assert chk["n_layers"] == nz and chk["height"] == h and chk["width"] == h, chk
    back = zarr.open(str(zpath), mode="r")["0"]
    assert np.array_equal(np.asarray(back[:, :64, :64]), block[:, :64, :64]), "round trip mismatch"
    print(f"verified: {chk['n_layers']} layers, {chk['px_um']} um, d = {chk['d_min']}..{chk['d_max']}")

    lv, sv = lab[y0:y1, x0:x1], sup[y0:y1, x0:x1]
    print(f"\nwrote {zpath}")
    print(f"      {a.name}__inklabels.npy   ink in {(lv > 0).mean() * 100:.2f}% of the crop")
    print(f"      {a.name}__supervision.npy supervised {(sv > 0).mean() * 100:.2f}% of the crop")
    print(f"      ink covers {(lv[sv > 0] > 0).mean() * 100:.1f}% of the supervised area")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
