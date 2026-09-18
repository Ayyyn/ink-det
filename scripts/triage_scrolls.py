#!/usr/bin/env python3
"""Stage A triage: locate each eligible scroll's axis, extent and wrap structure
from its published recto surface prediction, at the coarsest pyramid level.

Cheap by construction — the whole L5 array for an 8.64 um scroll is ~3 MB stored,
so all 23 First Letters volumes triage in a couple of minutes on a laptop with no
GPU and no volume download.

What this does NOT do: measure sheet separability. At L5 one voxel is ~276 um and a
papyrus sheet is ~150 um, so sheets are sub-voxel here. Separability needs L1/L2
windows and is stage B (score_windows.py). Stage A exists to find the umbilicus and
reject dead regions so stage B only streams windows worth paying for.

    python triage_scrolls.py --prize first-letters-2027 --px-max 8.64
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
logging.getLogger("aiobotocore").setLevel(logging.CRITICAL)
logging.getLogger("s3fs").setLevel(logging.CRITICAL)


def open_level(zarr_uri: str, level: int):
    """Open one pyramid level of an anonymous-S3 OME-Zarr as a zarr array."""
    import s3fs
    import zarr

    root = zarr_uri.replace("s3://", "").rstrip("/")
    fs = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(root=root, s3=fs, check=False)
    return zarr.open(store, mode="r")[str(level)]


def coarsest_level(zarr_uri: str, max_voxels: int = 120_000_000) -> tuple[int, tuple[int, ...]]:
    """Pick the finest pyramid level that still fits comfortably in RAM."""
    import s3fs
    import zarr

    root = zarr_uri.replace("s3://", "").rstrip("/")
    fs = s3fs.S3FileSystem(anon=True)
    g = zarr.open(s3fs.S3Map(root=root, s3=fs, check=False), mode="r")
    best = None
    for lvl in range(8):
        if str(lvl) not in g:
            continue
        shape = g[str(lvl)].shape
        if int(np.prod(shape)) <= max_voxels:
            best = (lvl, shape)
    if best is None:
        raise RuntimeError(f"no level under {max_voxels} voxels in {zarr_uri}")
    return best


def radial_profile(mask2d: np.ndarray, cy: float, cx: float, nbins: int) -> np.ndarray:
    """Mean occupancy as a function of radius from (cy, cx)."""
    ys, xs = np.nonzero(mask2d)
    if ys.size == 0:
        return np.zeros(nbins)
    r = np.hypot(ys - cy, xs - cx)
    rmax = r.max() if r.max() > 0 else 1.0
    idx = np.minimum((r / rmax * (nbins - 1)).astype(int), nbins - 1)
    hits = np.bincount(idx, minlength=nbins).astype(float)
    # normalise by the area of each annulus so a flat scroll reads flat
    edges = np.linspace(0, rmax, nbins + 1)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return hits / np.maximum(area, 1e-9)


def triage_one(scroll: str, volume: str, px_um: float, surf_uri: str) -> dict:
    lvl, shape = coarsest_level(surf_uri)
    a = open_level(surf_uri, lvl)
    vol = np.asarray(a[:]) > 0
    ds = 2 ** lvl
    vox_um = px_um * ds

    nz, ny, nx = vol.shape
    occ_z = vol.reshape(nz, -1).mean(axis=1)
    live = occ_z > (0.02 * occ_z.max() if occ_z.max() > 0 else 1)

    # umbilicus: per-slice centroid of surface voxels, over live slices only
    cys, cxs = [], []
    for z in np.nonzero(live)[0]:
        ys, xs = np.nonzero(vol[z])
        cys.append(ys.mean())
        cxs.append(xs.mean())
    cy = float(np.median(cys)) if cys else ny / 2
    cx = float(np.median(cxs)) if cxs else nx / 2
    # how much the axis wanders: a straight scroll is much easier to reason about
    axis_wobble_mm = (
        float(np.hypot(np.std(cys), np.std(cxs)) * vox_um / 1000.0) if cys else float("nan")
    )

    # radius from the axis to the outermost surface voxel, robust to specks
    ys, xs = np.nonzero(vol.any(axis=0))
    r_all = np.hypot(ys - cy, xs - cx) if ys.size else np.array([0.0])
    r_mm = float(np.percentile(r_all, 99) * vox_um / 1000.0)

    z_live = int(live.sum())
    height_mm = z_live * vox_um / 1000.0

    # Deliberately NOT estimating wrap count or recto area here. At this level one
    # voxel is ~276 um and a papyrus sheet is ~150 um, so wraps are sub-voxel: any
    # peak count is limited by radial binning (64 bins over ~45 mm = 0.7 mm/bin,
    # so >32 wraps is unrepresentable) rather than measured from the data, and an
    # area built on it understates the truth several-fold. Wrap spacing and recto
    # area are measured in stage B at L1/L2, where sheets are actually resolved.
    zmid = int(np.median(np.nonzero(live)[0])) if z_live else nz // 2

    return {
        "scroll": scroll,
        "volume": volume,
        "px_um": px_um,
        "level": lvl,
        "level_shape": list(shape),
        "voxel_um": round(vox_um, 1),
        "surface_fraction": round(float(vol.mean()), 4),
        "z_live": z_live,
        "z_total": nz,
        "height_mm": round(height_mm, 1),
        "radius_mm": round(r_mm, 1),
        "axis_wobble_mm": round(axis_wobble_mm, 2),
        "umbilicus_l0_yx": [round(cy * ds, 1), round(cx * ds, 1)],
        "mid_slice_z_level": zmid,
        "notes": "wrap count / recto area omitted: unresolvable at this pyramid level",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--assets", type=Path,
                    default=root / "scripts/eligible_assets_first-letters-2027.json")
    ap.add_argument("--villa", type=Path, default=root / "villa")
    ap.add_argument("--px-max", type=float, default=8.64,
                    help="only triage volumes at or below this voxel size")
    ap.add_argument("--only-new", action="store_true",
                    help="restrict to volumes not in the grand-prize-2027 list")
    ap.add_argument("--out", type=Path, default=root / "scripts/triage.json")
    args = ap.parse_args()

    assets = json.loads(args.assets.read_text(encoding="utf-8"))
    index = json.loads(
        (args.villa / "scrollprize.org/static/data_browser/index.json").read_text(encoding="utf-8")
    )
    by_id = {s["id"]: s for s in index["scrolls"]}

    if args.only_new:
        gp = json.loads(
            (args.villa / "scrollprize.org/src/data/prizeEligibility.json").read_text(
                encoding="utf-8")
        )["grand-prize-2027"]
        old = {e["scroll"] for e in gp}
        assets = [a for a in assets if a["scroll"] not in old]

    targets = [a for a in assets if a["px"] and a["px"] <= args.px_max]
    print(f"triaging {len(targets)} volumes at <= {args.px_max} um\n")

    rows = []
    for a in targets:
        s = by_id[a["scroll"]]
        surf = next(
            (p["zarr"] for p in (s.get("predictions") or [])
             if p.get("baseVolume") == a["volume"] and p["purpose"] == "surface-prediction"),
            None,
        )
        if surf is None:
            print(f"{a['scroll']:<11} no surface prediction for eligible volume — skipped")
            continue
        try:
            r = triage_one(a["scroll"], a["volume"], a["px"], surf)
        except Exception as exc:  # noqa: BLE001
            print(f"{a['scroll']:<11} FAILED: {type(exc).__name__}: {exc}")
            continue
        r["surface_zarr"] = surf
        rows.append(r)
        print(f"{r['scroll']:<11} L{r['level']} {str(r['level_shape']):<20} "
              f"{r['voxel_um']:>6.1f}um/vox  H {r['height_mm']:>6.1f}mm  "
              f"R {r['radius_mm']:>5.1f}mm  wobble {r['axis_wobble_mm']:>5.2f}mm  "
              f"surf_frac {r['surface_fraction']:.3f}")
        sys.stdout.flush()

    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  ({len(rows)} volumes)")

    if rows:
        print("\nranked by axis straightness (lower wobble = radial reasoning holds further):")
        for r in sorted(rows, key=lambda r: r["axis_wobble_mm"]):
            print(f"  {r['scroll']:<11} wobble {r['axis_wobble_mm']:>5.2f}mm   "
                  f"H {r['height_mm']:>5.0f}mm  R {r['radius_mm']:>4.1f}mm  "
                  f"surf_frac {r['surface_fraction']:.3f}")
        print()
        print("These are triage facts, not a target ranking. Sheet separability")
        print("decides the ranking, and that is stage B.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
