#!/usr/bin/env python3
"""Estimate the scroll axis (umbilicus) by converging the published lasagna normals.

`fit_spiral` needs an umbilicus, and Erwin Nieuwlaar used their own rather than the
centroid of the surface voxels. That choice is load-bearing: a Herculaneum scroll is a
crushed, non-circular spiral, so the centroid of its material is NOT the point the sheets
wrap around, and this project has already been burned once by assuming it was (the dead
region scoreboard, which measured a pitch that swung 242 -> 648 um purely with a
threshold). There is no automatic umbilicus tool in villa - VC3D expects a human to place
the control points - so this is the missing first step of the chain.

The estimator here is geometric rather than statistical. Every sheet is locally
perpendicular to its own normal, and concentric sheets wrap a common centre, so the
normals through the sheet voxels of one z-slab all very nearly pass through the axis.
For sample points p_i with unit normals n_i, the point minimising the summed squared
perpendicular distance to those lines is the solution of

    [ sum_i (I - n_i n_i^T) ] c = sum_i (I - n_i n_i^T) p_i

which is a 2x2 solve per slab. Where the centroid answers "where is the material", this
answers "what point do the sheets curve around" - the thing fit_spiral actually wants.

It reads the published lasagna nx/ny OME-Zarrs, so it costs no GPU and no volume
download. Normals are weighted by grad_mag, which is large on sheet surfaces and small in
the gaps, so the fit is dominated by actual sheet voxels.

**This is a starting axis, not a finished one.** Report the per-slab residual and
condition number it prints: a slab whose sheets are locally parallel (a flattened or
badly damaged region) is near-degenerate and its centre is poorly determined. Refine in
VC3D before a submission run, exactly as the winning entry did.

    python sprint/make_umbilicus.py --lasagna s3://vesuvius-challenge-open-data/PHerc0268/\\
        representations/predictions/lasagna/20251110183117-lasagna-20260419180421 \\
        --out work/umbilicus_PHerc0268.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def open_store(path: str):
    """Open a local or anonymous-S3 zarr group."""
    import zarr

    if path.startswith("s3://"):
        # let zarr build the store from the URL; wrapping a sync S3FileSystem by hand
        # gives a store that stalls on the first array access instead of failing
        return zarr.open(path, mode="r", storage_options={"anon": True})
    return zarr.open(path, mode="r")


def pick_group(root, want: str | None) -> tuple[str, float]:
    """Return (group, scale) for the lasagna store, reading the scale rather than assuming.

    The brief's hardest-won spiral gotcha: `lasagna_scale` must match the real downsample
    factor of the chosen group FOR THAT SCROLL. A mismatch silently reads wrong-resolution
    normals or throws an empty-z-ROI error, so it is read from .zattrs here and printed.
    """
    attrs = dict(root.attrs)
    ds = attrs["multiscales"][0]["datasets"]
    table = {d["path"]: float(d["coordinateTransformations"][0]["scale"][0]) for d in ds}
    if want is None:
        want = sorted(table, key=lambda k: table[k])[0]
    if want not in table:
        sys.exit(f"group {want} not in store; available: {table}")
    return want, table[want]


def accumulate(nx2d: np.ndarray, ny2d: np.ndarray, w2d: np.ndarray):
    """Add one slice's normal lines to the least-squares system, in true (y, x) coords.

    Returns (A, b, c0, n), where c0 is the constant term of the quadratic - without it the
    reported residual is meaningless (it can even come out negative). Summing these across the slices of a block and solving once is the
    same estimator as solving the block jointly, and keeps every position in its own
    slice's coordinate frame - stacking z into the row axis would corrupt the y the
    normals refer to.
    """
    m = w2d > 0
    n = int(m.sum())
    if not n:
        return np.zeros((2, 2)), np.zeros(2), 0.0, 0
    ys, xs = np.nonzero(m)
    p = np.stack([ys, xs], 1).astype(np.float64)
    d = np.stack([ny2d[m], nx2d[m]], 1).astype(np.float64)
    norm = np.linalg.norm(d, axis=1)
    ok = norm > 1e-6
    if not ok.any():
        return np.zeros((2, 2)), np.zeros(2), 0.0, 0
    p, d, wt = p[ok], d[ok] / norm[ok, None], w2d[m][ok].astype(np.float64)

    A = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            A[i, j] = np.sum(wt * ((1.0 if i == j else 0.0) - d[:, i] * d[:, j]))
    proj = p - d * np.sum(d * p, axis=1, keepdims=True)      # (I - dd^T) p
    b = (wt[:, None] * proj).sum(0)
    c0 = float(np.sum(wt * np.sum(proj * p, axis=1)))
    return A, b, c0, int(ok.sum())


def solve_centre(A: np.ndarray, b: np.ndarray, c0: float, wsum: float, n: int, min_samples: int):
    """Solve the accumulated system; reject the degenerate (locally parallel sheets) case."""
    if n < min_samples:
        return np.nan, np.nan, np.nan, np.nan, n
    cond = float(np.linalg.cond(A))
    if not np.isfinite(cond) or cond > 1e6:
        return np.nan, np.nan, np.nan, cond, n
    c = np.linalg.solve(A, b)
    # weighted RMS perpendicular miss distance, in level-group pixels. The full quadratic
    # is c.A.c - 2 c.b + c0; dropping c0 makes this ~0 regardless of fit quality.
    resid = float(np.sqrt(max(0.0, (c @ A @ c - 2 * c @ b + c0) / max(wsum, 1e-9))))
    return float(c[0]), float(c[1]), resid, cond, n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lasagna", required=True, help="lasagna prediction dir (local or s3://)")
    ap.add_argument("--scroll", default=None, help="name prefix; inferred from the path if omitted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", default=None, help="OME-Zarr group; default is the finest")
    ap.add_argument("--step", type=int, default=160,
                    help="z spacing of control points in LEVEL-0 voxels (default matches "
                         "the winning entry's umbilicus)")
    ap.add_argument("--grad-percentile", type=float, default=90.0,
                    help="keep voxels above this grad_mag percentile; these are the sheets")
    ap.add_argument("--min-samples", type=int, default=500)
    ap.add_argument("--max-slabs", type=int, default=0, help="0 = all; smaller is a quick look")
    ap.add_argument("--z0", type=int, default=None,
                    help="restrict to this z range, in LEVEL-0 voxels. Survey the whole "
                         "scroll at a coarse group to choose a band, then refine over just "
                         "that band at a fine group - the fine group over the whole scroll "
                         "is tens of GB of normals for z you will never fit.")
    ap.add_argument("--z1", type=int, default=None)
    a = ap.parse_args(argv)

    base = a.lasagna.rstrip("/")
    scroll = a.scroll or next((p for p in base.split("/") if p.startswith("PHerc")), "scroll")
    stores = {k: open_store(f"{base}/{scroll}_{k}.ome.zarr") for k in ("nx", "ny", "grad_mag")}
    group, scale = pick_group(stores["nx"], a.group)
    print(f"{scroll}: lasagna group {group} at scale {scale:g}  "
          f"(this is the lasagna_scale fit_spiral must be given)")

    nxa, nya, gma = (stores[k][group] for k in ("nx", "ny", "grad_mag"))
    nz, ny_, nx_ = nxa.shape
    print(f"  volume {nz} x {ny_} x {nx_} at level scale {scale:g}", flush=True)

    # Read CHUNK-ALIGNED z blocks, not single slices. These stores are chunked 32x32x32,
    # so one z-plane touches ~576 chunks and takes ~28 s over S3 while a whole 32-slice
    # block costs one chunk-row. Aggregating the block also gives each centre far more
    # samples, which is what the least-squares solve wants anyway.
    cz = int(getattr(nxa, "chunks", (32,))[0]) or 32
    block = max(cz, int(round(a.step / scale / cz)) * cz) if a.step else cz
    starts = list(range(0, nz, block))
    if a.z0 is not None or a.z1 is not None:
        # The band is given in level-0 voxels; these stores are indexed in group pixels.
        g0 = int((a.z0 or 0) / scale)
        g1 = int((a.z1 if a.z1 is not None else nz * scale) / scale)
        starts = [z for z in starts if z + block > g0 and z < g1]
        if not starts:
            sys.exit(f"z range {a.z0}..{a.z1} is empty at group {group} "
                     f"(store is {nz} slices x scale {scale:g} = {nz * scale:.0f} level-0)")
        print(f"  restricted to z {a.z0}..{a.z1} level-0 = group slices "
              f"{starts[0]}..{min(starts[-1] + block, nz)}", flush=True)
    if a.max_slabs:
        starts = starts[:a.max_slabs]
    print(f"  {len(starts)} blocks of {block} slices "
          f"(= {block * scale:.0f} level-0 voxels per control point)", flush=True)

    pts, bad = [], 0
    for i, z0 in enumerate(starts):
        z1 = min(z0 + block, nz)
        gm = np.asarray(gma[z0:z1]).astype(np.float32)
        if gm.max() <= 0:
            bad += 1
            continue
        thr = np.percentile(gm[gm > 0], a.grad_percentile)
        w = np.where(gm >= max(thr, 1e-9), gm, 0)
        nxb, nyb = np.asarray(nxa[z0:z1]), np.asarray(nya[z0:z1])
        A = np.zeros((2, 2)); bvec = np.zeros(2); c0 = 0.0; ntot = 0
        for k in range(z1 - z0):
            Ak, bk, c0k, nk = accumulate(nxb[k], nyb[k], w[k])
            A += Ak; bvec += bk; c0 += c0k; ntot += nk
        cy, cx, resid, cond, n = solve_centre(A, bvec, c0, float(w.sum()), ntot, a.min_samples)
        zc = (z0 + z1) / 2
        if not np.isfinite(cy):
            bad += 1
            print(f"  z {zc * scale:8.0f}  UNDETERMINED (n={n}, cond={cond:.3g})", flush=True)
            continue
        pts.append({"x": round(cx * scale, 1), "y": round(cy * scale, 1),
                    "z": int(round(zc * scale)), "score": 100,
                    "_resid_px": round(resid, 1), "_cond": round(cond, 2), "_n": n})
        print(f"  z {zc * scale:8.0f}  centre ({cx * scale:7.1f}, {cy * scale:7.1f})  "
              f"resid {resid * scale:6.1f} vox  cond {cond:5.2f}  n {n}", flush=True)

    if not pts:
        sys.exit("no slab produced a usable centre; check the lasagna path and --group")

    # A real scroll axis is close to straight. A wandering fit means the estimator is being
    # pulled around by damage, and is the signal to fix it by hand in VC3D.
    xy = np.array([[p["x"], p["y"]] for p in pts])
    drift = float(np.hypot(*(xy.max(0) - xy.min(0))))
    spread = float(np.mean(np.linalg.norm(xy - xy.mean(0), axis=1)))
    print(f"\n{len(pts)} control points, {bad} slab(s) undetermined")
    print(f"  axis wander: {drift:.0f} level-0 voxels end to end, mean radius {spread:.0f}")
    if drift > 400:
        print("  *** the axis wanders a lot. Inspect in VC3D before fitting; a bad umbilicus")
        print("      is the failure mode that quietly produces a plausible but wrong spiral.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"control_points": pts}, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    print("  underscore-prefixed fields are diagnostics; villa reads only x, y, z, score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
