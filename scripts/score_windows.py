#!/usr/bin/env python3
"""Stage B: score candidate 4 cm2 windows by how well adjacent papyrus wraps stay
separated, straight off the published recto surface prediction.

Why this metric. The 2026 open-problems writeup names the compressed-region problem
-- "some regions lose effective separability between layers" -- as a top bottleneck,
and asks for scan-quality metrics. A patch is growable when a radial ray out from the
scroll axis crosses sheet, gap, sheet, gap at a regular pitch. It is hopeless when
wraps merge into blobs or the prediction goes hollow. Both are measurable with no
labels, no ground truth and no GPU.

Level choice matters. At L5 (~276 um/voxel) a papyrus sheet is sub-voxel and wrap
counts are pure binning artefact. This runs at L3 (~69 um/voxel for an 8.64 um
scroll), where a 300-600 um wrap pitch spans 4-9 voxels and is genuinely resolved.
Treat the output as a ranking to be confirmed at L1/L2 on the top few windows.

Output is a ranked CSV/JSON of windows with L0 (x, y, z) centres, ready to use as
GrowPatch seed points in VC3D.

    python score_windows.py --scroll PHerc0343
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
for _n in ("aiobotocore", "s3fs", "botocore", "aiohttp"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

# A 4 cm2 First Letters patch is 20 mm x 20 mm.
PATCH_MM = 20.0


def open_level(zarr_uri: str, level: int):
    import s3fs
    import zarr

    root = zarr_uri.replace("s3://", "").rstrip("/")
    fs = s3fs.S3FileSystem(anon=True)
    return zarr.open(s3fs.S3Map(root=root, s3=fs, check=False), mode="r")[str(level)]


def ray_metrics(mask2d: np.ndarray, cy: float, cx: float, angles: np.ndarray,
                radii_vox: np.ndarray) -> dict[str, np.ndarray]:
    """Cast rays from (cy, cx) and measure wrap structure along each.

    Returns per-(angle, radial-band) arrays. Fully vectorised: transitions are found
    on the whole (n_angles, n_radii) sample grid at once, then grouped by ray.
    """
    ny, nx = mask2d.shape
    # sample grid: (n_angles, n_radii)
    yy = cy + np.sin(angles)[:, None] * radii_vox[None, :]
    xx = cx + np.cos(angles)[:, None] * radii_vox[None, :]
    inb = (yy >= 0) & (yy < ny - 1) & (xx >= 0) & (xx < nx - 1)
    yi = np.clip(yy.astype(np.int32), 0, ny - 1)
    xi = np.clip(xx.astype(np.int32), 0, nx - 1)
    samp = mask2d[yi, xi] & inb  # (n_angles, n_radii) bool

    n_ang, n_rad = samp.shape
    # transitions along the radial axis
    d = np.diff(samp.astype(np.int8), axis=1)
    up_a, up_r = np.nonzero(d == 1)    # background -> sheet (entering a wrap)
    dn_a, dn_r = np.nonzero(d == -1)   # sheet -> background (leaving a wrap)

    out = {
        "n_cross": np.zeros((n_ang,), np.float32),
        "gap_mean": np.full((n_ang,), np.nan, np.float32),
        "gap_cv": np.full((n_ang,), np.nan, np.float32),
        "gap_max": np.full((n_ang,), np.nan, np.float32),
        "sheet_mean": np.full((n_ang,), np.nan, np.float32),
        "occupancy": samp.mean(axis=1).astype(np.float32),
    }
    if up_a.size == 0:
        return out

    out["n_cross"] = np.bincount(up_a, minlength=n_ang).astype(np.float32)

    # gaps = distance between consecutive entries into a wrap, within the same ray
    order = np.lexsort((up_r, up_a))
    a_s, r_s = up_a[order], up_r[order]
    same = a_s[1:] == a_s[:-1]
    gaps = (r_s[1:] - r_s[:-1])[same].astype(np.float64)
    gap_ray = a_s[1:][same]
    if gaps.size:
        cnt = np.bincount(gap_ray, minlength=n_ang).astype(np.float64)
        s1 = np.bincount(gap_ray, weights=gaps, minlength=n_ang)
        s2 = np.bincount(gap_ray, weights=gaps ** 2, minlength=n_ang)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s1 / cnt
            var = np.maximum(s2 / cnt - mean ** 2, 0.0)
            cv = np.sqrt(var) / mean
        ok = cnt > 0
        out["gap_mean"][ok] = mean[ok].astype(np.float32)
        out["gap_cv"][ok] = cv[ok].astype(np.float32)
        gmax = np.zeros(n_ang)
        np.maximum.at(gmax, gap_ray, gaps)
        out["gap_max"][ok] = gmax[ok].astype(np.float32)

    # sheet thickness = run length of each wrap crossing, paired within a ray
    order_d = np.lexsort((dn_r, dn_a))
    da_s, dr_s = dn_a[order_d], dn_r[order_d]
    # pair each 'up' with the next 'down' on the same ray
    thick, thick_ray = [], []
    # group boundaries per ray for both sets
    for arr_a, arr_r, other_a, other_r in ((a_s, r_s, da_s, dr_s),):
        starts = np.searchsorted(other_a, arr_a, side="left")
        ends = np.searchsorted(other_a, arr_a, side="right")
        for k in range(arr_a.size):
            lo, hi = starts[k], ends[k]
            if hi <= lo:
                continue
            seg = other_r[lo:hi]
            j = np.searchsorted(seg, arr_r[k], side="right")
            if j < seg.size:
                thick.append(seg[j] - arr_r[k])
                thick_ray.append(arr_a[k])
    if thick:
        t = np.asarray(thick, np.float64)
        tr = np.asarray(thick_ray)
        cnt = np.bincount(tr, minlength=n_ang).astype(np.float64)
        s1 = np.bincount(tr, weights=t, minlength=n_ang)
        ok = cnt > 0
        out["sheet_mean"][ok] = (s1[ok] / cnt[ok]).astype(np.float32)
    return out


def score_scroll(scroll: str, surf_uri: str, px_um: float, level: int,
                 n_angles: int, z_step_mm: float, band_mm: float,
                 r_min_mm: float) -> tuple[list[dict], dict]:
    a = open_level(surf_uri, level)
    nz, ny, nx = a.shape
    ds = 2 ** level
    vox_um = px_um * ds
    vox_mm = vox_um / 1000.0

    z_step = max(1, int(round(z_step_mm / vox_mm)))
    chunk_z = a.chunks[0]
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    r_max_vox = int(min(ny, nx) // 2) - 2
    radii = np.arange(0, r_max_vox, 1.0)

    # radial bands: a window sits on one wrap, so score within a band of radii
    band_vox = max(2, int(round(band_mm / vox_mm)))
    r_min_vox = int(round(r_min_mm / vox_mm))
    band_edges = np.arange(r_min_vox, r_max_vox, band_vox)

    print(f"  L{level}  {vox_um:.1f} um/vox  shape {a.shape}  chunk_z {chunk_z}")
    print(f"  {n_angles} angles, z every {z_step} vox ({z_step*vox_mm:.1f} mm), "
          f"{len(band_edges)} radial bands of {band_mm} mm")

    recs: list[dict] = []
    for z0 in range(0, nz, chunk_z):
        z1 = min(z0 + chunk_z, nz)
        slab = np.asarray(a[z0:z1]) > 0
        if not slab.any():
            continue
        for zl in range(0, z1 - z0, z_step):
            m = slab[zl]
            if m.sum() < 50:
                continue
            ys, xs = np.nonzero(m)
            cy, cx = ys.mean(), xs.mean()  # per-slice centroid tracks axis wobble
            for b0 in band_edges:
                b1 = min(b0 + band_vox, r_max_vox)
                rr = radii[(radii >= b0) & (radii < b1)]
                if rr.size < 4:
                    continue
                mm = ray_metrics(m, cy, cx, angles, rr)
                recs.append({
                    "z": z0 + zl,
                    "r_lo_vox": int(b0),
                    "r_hi_vox": int(b1),
                    "cy": float(cy), "cx": float(cx),
                    "per_angle": mm,
                })
        print(f"    z {z0}-{z1}  cells so far {len(recs)}", flush=True)

    meta = {
        "scroll": scroll, "level": level, "vox_um": vox_um,
        "n_angles": n_angles, "band_vox": band_vox, "z_step_vox": z_step,
        "shape": list(a.shape), "downsample": ds, "px_um": px_um,
    }
    return recs, meta


def aggregate_windows(recs: list[dict], meta: dict) -> list[dict]:
    """Aggregate per-(z, band, angle) cells into 4 cm2 windows and score them."""
    vox_mm = meta["vox_um"] / 1000.0
    ds = meta["downsample"]
    n_ang = meta["n_angles"]
    z_step = meta["z_step_vox"]

    # how many sampled z rows make up 20 mm, and how many angles make 20 mm of arc
    z_rows = max(1, int(round(PATCH_MM / vox_mm / z_step)))

    by_band: dict[tuple[int, int], list[dict]] = {}
    for r in recs:
        by_band.setdefault((r["r_lo_vox"], r["r_hi_vox"]), []).append(r)

    out: list[dict] = []
    for (b0, b1), cells in by_band.items():
        cells.sort(key=lambda c: c["z"])
        r_mid_mm = (b0 + b1) / 2 * vox_mm
        if r_mid_mm <= 0:
            continue
        # 20 mm of arc at this radius, in angular samples
        dtheta = PATCH_MM / r_mid_mm
        n_ang_win = max(1, int(round(dtheta / (2 * np.pi) * n_ang)))
        if n_ang_win > n_ang:
            continue

        # stack this band's metrics into (n_z, n_ang) arrays
        zs = np.array([c["z"] for c in cells])
        stack = {k: np.vstack([c["per_angle"][k] for c in cells])
                 for k in ("n_cross", "gap_mean", "gap_cv", "gap_max",
                           "sheet_mean", "occupancy")}

        n_z = len(cells)
        for zi in range(0, max(1, n_z - z_rows + 1)):
            for ai in range(0, n_ang, max(1, n_ang_win // 2)):
                sel_a = (np.arange(ai, ai + n_ang_win) % n_ang)
                sl = (slice(zi, zi + z_rows), sel_a)

                n_cross = stack["n_cross"][sl]
                gap_cv = stack["gap_cv"][sl]
                gap_mean = stack["gap_mean"][sl]
                gap_max = stack["gap_max"][sl]
                sheet = stack["sheet_mean"][sl]
                occ = stack["occupancy"][sl]

                valid = np.isfinite(gap_cv)
                # need most of the window to actually have measurable wrap structure
                if valid.mean() < 0.6:
                    continue

                cv = float(np.nanmedian(gap_cv))
                gm = float(np.nanmedian(gap_mean))
                gx = float(np.nanmedian(gap_max))
                sh = float(np.nanmedian(sheet))
                oc = float(np.mean(occ))
                nc = float(np.median(n_cross))

                # --- composite, each term in [0, 1], higher is better ---
                # regular pitch: CV of 0 is perfect, 1.0 is noise
                regularity = float(np.clip(1.0 - cv, 0.0, 1.0))
                # thin sheets: a wrap should be ~1-3 voxels at this level, not a blob
                thinness = float(np.clip(1.0 - (sh - 2.0) / 6.0, 0.0, 1.0))
                # no hollow stretch: worst gap shouldn't dwarf the typical pitch
                solidity = float(np.clip(1.0 - (gx / max(gm, 1e-6) - 1.0) / 4.0, 0.0, 1.0))
                # enough wraps crossed to be a real layered region
                layered = float(np.clip(nc / 8.0, 0.0, 1.0))
                score = regularity * thinness * solidity * layered

                z_c = int(zs[zi] + (zs[min(zi + z_rows - 1, n_z - 1)] - zs[zi]) / 2)
                th_c = float(((ai + n_ang_win / 2) % n_ang) / n_ang * 2 * np.pi)
                r_c = (b0 + b1) / 2
                cy = float(np.mean([c["cy"] for c in cells[zi:zi + z_rows]]))
                cx = float(np.mean([c["cx"] for c in cells[zi:zi + z_rows]]))

                out.append({
                    "scroll": meta["scroll"],
                    "score": round(score, 4),
                    "regularity": round(regularity, 3),
                    "thinness": round(thinness, 3),
                    "solidity": round(solidity, 3),
                    "layered": round(layered, 3),
                    "gap_cv": round(cv, 3),
                    "gap_mean_vox": round(gm, 2),
                    "gap_mean_um": round(gm * meta["vox_um"], 1),
                    "sheet_mean_vox": round(sh, 2),
                    "n_cross_median": round(nc, 1),
                    "occupancy": round(oc, 4),
                    "r_mm": round(r_c * vox_mm, 1),
                    "theta_deg": round(np.degrees(th_c), 1),
                    # L0 coordinates, for pasting into VC3D as a GrowPatch seed
                    "x_l0": int(round((cx + np.cos(th_c) * r_c) * ds)),
                    "y_l0": int(round((cy + np.sin(th_c) * r_c) * ds)),
                    "z_l0": int(round(z_c * ds)),
                })
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", action="append", required=True,
                    help="scroll id; repeatable")
    ap.add_argument("--triage", type=Path, default=root / "scripts/triage.json")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--n-angles", type=int, default=360)
    ap.add_argument("--z-step-mm", type=float, default=2.0)
    ap.add_argument("--band-mm", type=float, default=5.0)
    ap.add_argument("--r-min-mm", type=float, default=6.0)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", type=Path, default=root / "scripts/windows.json")
    args = ap.parse_args()

    triage = {r["scroll"]: r for r in
              json.loads(args.triage.read_text(encoding="utf-8"))}

    all_win: list[dict] = []
    for scroll in args.scroll:
        if scroll not in triage:
            print(f"{scroll}: not in {args.triage.name} — run triage_scrolls.py first")
            continue
        t = triage[scroll]
        print(f"\n=== {scroll} ===")
        recs, meta = score_scroll(
            scroll, t["surface_zarr"], t["px_um"], args.level,
            args.n_angles, args.z_step_mm, args.band_mm, args.r_min_mm,
        )
        wins = aggregate_windows(recs, meta)
        print(f"  {len(wins)} scored windows")
        all_win.extend(wins)
        sys.stdout.flush()

    all_win.sort(key=lambda w: -w["score"])
    args.out.write_text(json.dumps(all_win[:5000], indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  ({len(all_win)} windows, top 5000 kept)")

    print(f"\ntop {args.top} candidate 4 cm2 windows:")
    hdr = (f"{'scroll':<11} {'score':>6} {'reg':>5} {'thin':>5} {'sol':>5} {'lay':>5} "
           f"{'pitch_um':>8} {'r_mm':>5} {'theta':>6} {'x_l0':>7} {'y_l0':>7} {'z_l0':>7}")
    print(hdr); print("-" * len(hdr))
    for w in all_win[:args.top]:
        print(f"{w['scroll']:<11} {w['score']:>6.3f} {w['regularity']:>5.2f} "
              f"{w['thinness']:>5.2f} {w['solidity']:>5.2f} {w['layered']:>5.2f} "
              f"{w['gap_mean_um']:>8.0f} {w['r_mm']:>5.1f} {w['theta_deg']:>6.1f} "
              f"{w['x_l0']:>7} {w['y_l0']:>7} {w['z_l0']:>7}")

    if all_win:
        print("\nper-scroll best:")
        seen = set()
        for w in all_win:
            if w["scroll"] in seen:
                continue
            seen.add(w["scroll"])
            print(f"  {w['scroll']:<11} best score {w['score']:.3f}  "
                  f"pitch {w['gap_mean_um']:.0f} um  seed L0 "
                  f"({w['x_l0']}, {w['y_l0']}, {w['z_l0']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
