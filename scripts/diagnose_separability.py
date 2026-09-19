#!/usr/bin/env python3
"""Two diagnostics on the wrap-separability measurement, before anyone trusts a
ranking built from it.

1. THRESHOLD SENSITIVITY. Scoring drops from 2110 windows to 97 when the
   despeckle threshold goes from off to 20 px, and the top-ranked window moves
   across the scroll. If pitch and CV swing as hard, the ranking is an artefact of
   an arbitrary constant and should not be shipped.

2. POISSON NULL. The pitch CV sits near 0.5 everywhere and does not move when
   speckle is filtered out. Two very different worlds produce that:

     (a) the scroll really is wound irregularly, or
     (b) the surface prediction has no clean concentric structure at this scale,
         and radial crossings are closer to a random point process than a spiral.

   A Poisson process has exponentially distributed gaps and CV = 1.0; perfectly
   regular wraps give CV = 0. Measuring CV alone cannot separate "somewhat
   irregular spiral" from "structured noise". The autocorrelation of the radial
   signal can: a periodic spiral shows a clear peak at lag = pitch, a random
   process shows none. Comparing the observed ACF peak against a matched-density
   Poisson null is the decisive test.

   This matters for what gets submitted. Outcome (a) means we have a scan-quality
   metric. Outcome (b) means we have evidence the 9 um surface model is failing to
   recover concentric structure on unread scrolls -- a core-pipeline finding.

    python diagnose_separability.py --scroll PHerc0343
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
for _n in ("aiobotocore", "s3fs", "botocore", "aiohttp"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

from score_windows import despeckle, open_level  # noqa: E402


def ray_signals(mask2d, cy, cx, n_angles, r0, r1, step=0.25):
    """Binary radial signal per ray, sampled every `step` voxels.

    Stepping a full voxel along a ray while indexing a Cartesian grid with nearest
    neighbour aliases: the sampled sequence picks up a 2-sample periodicity that is
    a property of the sampling lattice, not of the papyrus. That artefact sits
    exactly at the Nyquist limit and swamps the autocorrelation. Sub-voxel steps
    push the lattice artefact well above the frequencies of interest, so a real
    300-600 um wrap pitch (4-9 voxels, i.e. 16-36 samples at step 0.25) is
    resolved cleanly.
    """
    ang = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    rad = np.arange(r0, r1, step)
    ny, nx = mask2d.shape
    yy = cy + np.sin(ang)[:, None] * rad[None, :]
    xx = cx + np.cos(ang)[:, None] * rad[None, :]
    inb = (yy >= 0) & (yy < ny - 1) & (xx >= 0) & (xx < nx - 1)
    return mask2d[np.clip(np.rint(yy).astype(np.int32), 0, ny - 1),
                  np.clip(np.rint(xx).astype(np.int32), 0, nx - 1)] & inb


def gap_stats(sig: np.ndarray) -> tuple[float, float, float]:
    """Median pitch, pooled CV of gaps, and mean crossings per ray."""
    d = np.diff(sig.astype(np.int8), axis=1)
    a, r = np.nonzero(d == 1)
    if a.size < 2:
        return float("nan"), float("nan"), 0.0
    order = np.lexsort((r, a))
    a_s, r_s = a[order], r[order]
    same = a_s[1:] == a_s[:-1]
    gaps = (r_s[1:] - r_s[:-1])[same].astype(float)
    if gaps.size < 2:
        return float("nan"), float("nan"), a.size / sig.shape[0]
    return float(np.median(gaps)), float(gaps.std() / gaps.mean()), a.size / sig.shape[0]


def acf_peak(sig: np.ndarray, max_lag: int,
             lag_lo: int, lag_hi: int) -> tuple[np.ndarray, float, int]:
    """Mean radial autocorrelation, and the prominence of its best peak within the
    lag band where a real wrap pitch must lie.

    Two things have to be right for this to mean anything. First, the search has to
    be confined to physically plausible pitches -- an unconstrained argmax over a
    monotonically decaying ACF just returns the first bin, which looks like a
    detection but is only the decay. Second, prominence must be measured against
    the local decay, not against a global median: the ACF of any blobby signal
    falls off smoothly, and that falloff would otherwise be scored as signal. So
    the ACF is detrended with a wide uniform filter first, and the peak is measured
    on the residual.
    """
    x = sig.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    # FFT-based autocorrelation, zero padded to avoid circular wrap
    f = np.fft.rfft(x, n=2 * n, axis=1)
    ac = np.fft.irfft(f * np.conj(f), n=2 * n, axis=1)[:, :max_lag]
    denom = ac[:, :1].copy()
    denom[denom == 0] = 1.0
    ac = (ac / denom).mean(axis=0)

    lag_hi = min(lag_hi, max_lag - 1)
    if lag_hi - lag_lo < 3:
        return ac, 0.0, 0

    # detrend: subtract a smoothed copy so only oscillation survives
    from scipy.ndimage import uniform_filter1d

    win = max(3, (lag_hi - lag_lo) // 2 | 1)
    resid = ac - uniform_filter1d(ac, size=win, mode="nearest")

    seg = resid[lag_lo:lag_hi]
    k = int(np.argmax(seg)) + lag_lo
    # prominence relative to the scatter of the residual, so it is comparable
    # across scrolls with different densities
    scale = float(np.std(resid[lag_lo:lag_hi])) or 1.0
    return ac, float(resid[k] / scale), k


def poisson_null(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random binary signal matched to the observed crossing count and run length."""
    n_ang, n_r = sig.shape
    d = np.diff(sig.astype(np.int8), axis=1)
    n_cross = int((d == 1).sum())
    if n_cross == 0:
        return np.zeros_like(sig)
    runs = int(max(1, round(sig.sum() / max(n_cross, 1))))
    per_ray = max(1, n_cross // n_ang)
    out = np.zeros_like(sig)
    for i in range(n_ang):
        starts = rng.integers(0, max(1, n_r - runs), size=per_ray)
        for s in starts:
            out[i, s:s + runs] = True
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", default="PHerc0343")
    ap.add_argument("--surface-uri", default=None,
                    help="explicit surface-prediction zarr; bypasses triage.json so a "
                         "READ scroll can be used as a positive control")
    ap.add_argument("--px-um", type=float, default=None)
    ap.add_argument("--triage", type=Path, default=root / "scripts/triage.json")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--n-angles", type=int, default=360)
    ap.add_argument("--n-slices", type=int, default=6)
    ap.add_argument("--r-min-mm", type=float, default=6.0)
    ap.add_argument("--thresholds", type=int, nargs="+", default=[1, 5, 10, 20, 40])
    ap.add_argument("--pitch-lo-um", type=float, default=150.0)
    ap.add_argument("--pitch-hi-um", type=float, default=1500.0)
    ap.add_argument("--step", type=float, default=0.25,
                    help="radial sampling step in voxels; <1 avoids lattice aliasing")
    args = ap.parse_args()

    if args.surface_uri:
        surf_uri, px_um = args.surface_uri, args.px_um
        if px_um is None:
            raise SystemExit("--px-um is required with --surface-uri")
    else:
        tri = {r["scroll"]: r for r in
               json.loads(args.triage.read_text(encoding="utf-8"))}
        t = tri[args.scroll]
        surf_uri, px_um = t["surface_zarr"], t["px_um"]
    a = open_level(surf_uri, args.level)
    vox_um = px_um * (2 ** args.level)
    nz, ny, nx = a.shape
    r0 = int(round(args.r_min_mm * 1000 / vox_um))
    r1 = int(min(ny, nx) // 2) - 2

    # sample slices spread through the live part of the scroll, chunk-aligned reads
    zs = np.linspace(nz * 0.2, nz * 0.8, args.n_slices).astype(int)
    print(f"{args.scroll}  L{args.level}  {vox_um:.1f} um/vox  slices {list(zs)}")
    print(f"radii {r0}-{r1} vox ({r0*vox_um/1000:.1f}-{r1*vox_um/1000:.1f} mm)\n")

    slices = []
    for z in zs:
        m = np.asarray(a[int(z)]) > 0
        if m.sum() > 50:
            slices.append(m)
    if not slices:
        print("no populated slices")
        return 1

    print("=== 1. threshold sensitivity ===")
    hdr = f"{'min_px':>7} {'pitch_um':>9} {'gap_CV':>7} {'cross/ray':>10} {'kept_frac':>10}"
    print(hdr); print("-" * len(hdr))
    keep_ref = None
    for mp in args.thresholds:
        pit, cvs, crs, kept = [], [], [], []
        for m in slices:
            mf = despeckle(m, mp)
            kept.append(mf.sum() / max(m.sum(), 1))
            ys, xs = np.nonzero(mf)
            if ys.size < 50:
                continue
            sig = ray_signals(mf, ys.mean(), xs.mean(), args.n_angles, r0, r1, args.step)
            p, c, k = gap_stats(sig)
            if np.isfinite(p):
                pit.append(p); cvs.append(c); crs.append(k)
        if not pit:
            print(f"{mp:>7} {'--':>9} {'--':>7} {'--':>10} {np.mean(kept):>10.3f}")
            continue
        if keep_ref is None:
            keep_ref = np.median(pit)
        print(f"{mp:>7} {np.median(pit)*args.step*vox_um:>9.0f} {np.median(cvs):>7.3f} "
              f"{np.mean(crs):>10.1f} {np.mean(kept):>10.3f}")

    print("\n=== 2. Poisson null: is there periodic concentric structure? ===")
    print("CV alone cannot tell an irregular spiral from structured noise.")
    print("A real spiral puts a peak in the radial autocorrelation at lag = pitch.\n")
    rng = np.random.default_rng(0)
    hdr2 = (f"{'min_px':>7} {'obs_peak':>9} {'null_peak':>10} {'ratio':>7} "
            f"{'obs_lag_um':>11} {'lag_vox':>9} {'verdict':>22}")
    print(hdr2); print("-" * len(hdr2))
    for mp in args.thresholds:
        op, npk, lags = [], [], []
        for m in slices:
            mf = despeckle(m, mp)
            ys, xs = np.nonzero(mf)
            if ys.size < 50:
                continue
            sig = ray_signals(mf, ys.mean(), xs.mean(), args.n_angles, r0, r1, args.step)
            max_lag = min(sig.shape[1] // 2, int(200 / args.step))
            # plausible wrap pitch band, converted to sample lags
            lag_lo = max(3, int(args.pitch_lo_um / vox_um / args.step))
            lag_hi = int(args.pitch_hi_um / vox_um / args.step)
            _, pk, lag = acf_peak(sig, max_lag, lag_lo, lag_hi)
            _, pk0, _ = acf_peak(poisson_null(sig, rng), max_lag, lag_lo, lag_hi)
            op.append(pk); npk.append(pk0); lags.append(lag)
        if not op:
            continue
        o, n0 = float(np.mean(op)), float(np.mean(npk))
        ratio = o / n0 if n0 > 1e-9 else float("inf")
        if o > 2.5 and ratio > 1.5:
            v = "periodic structure"
        elif o > 1.5:
            v = "weak periodicity"
        else:
            v = "indistinguishable"
        lag_vox = np.median(lags) * args.step
        print(f"{mp:>7} {o:>9.4f} {n0:>10.4f} {ratio:>7.2f} "
              f"{lag_vox*vox_um:>11.0f} {lag_vox:>9.2f} {v:>22}")

    print("\nReading it: ratio >3 means real concentric wraps the metric can rank.")
    print("Near 1 means the prediction has no periodic radial structure here, and")
    print("the separability score is measuring the surface model, not the scroll.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
