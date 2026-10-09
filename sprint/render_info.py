#!/usr/bin/env python3
"""Report a surface volume's geometry, and the depth range its layer count allows.

Hardcoding this is how the first PHerc0800 run broke: the scripts assumed Erwin
Nieuwlaar's 66-layer renders, the published ones are 31, and the positive control
on PHerc0139 w035 is 28 at a different pixel size again. Every caller should ask
the render rather than assume.

The villa ink models consume a 17-layer window s..s+16 centred on layer N/2 + d, so

    s = N//2 + d - 8     and     0 <= s  and  s + 17 <= N

which bounds d to [-(N//2 - 8), N - 17 - (N//2 - 8)]. Hecate centres on (N-1)/2 + d
and needs the render's true pixel size to resample onto its 9.6 um grid: passing
8.64 for a 9.362 um scroll silently resamples to the wrong scale.

    python sprint/render_info.py <render.zarr>            # human readable
    eval "$(python sprint/render_info.py <render.zarr> --sh)"   # N_LAYERS, PX_UM, D_MIN, D_MAX
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def info(path: str) -> dict:
    """Read shape and pixel size from a local or s3:// OME-Zarr surface volume."""
    if path.startswith("s3://"):
        import s3fs

        fs = s3fs.S3FileSystem(anon=True)
        root = path[5:].rstrip("/")
        read = lambda rel: fs.cat(f"{root}/{rel}")
        exists = lambda rel: fs.exists(f"{root}/{rel}")
    else:
        root = Path(path)
        read = lambda rel: (root / rel).read_bytes()
        exists = lambda rel: (root / rel).exists()

    for rel in ("0/.zarray", ".zarray", "0/zarr.json", "zarr.json"):
        if exists(rel):
            arr = json.loads(read(rel))
            break
    else:
        raise FileNotFoundError(f"no zarr array metadata under {path}")
    nz, ny, nx = (int(v) for v in arr["shape"])

    px = None
    if exists(".zattrs"):
        try:
            ms = json.loads(read(".zattrs")).get("multiscales") or []
            for ds in (ms[0].get("datasets") if ms else []) or []:
                if str(ds.get("path")) == "0":
                    for ct in ds.get("coordinateTransformations", []):
                        if ct.get("type") == "scale":
                            px = float(ct["scale"][-1])
        except Exception:
            px = None

    half = nz // 2
    return {"n_layers": nz, "height": ny, "width": nx, "px_um": px,
            "d_min": -(half - 8), "d_max": nz - 17 - (half - 8),
            "mid": half, "hecate_zc0": (nz - 1) / 2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("render")
    ap.add_argument("--sh", action="store_true", help="emit shell assignments for eval")
    a = ap.parse_args(argv)

    d = info(a.render)
    if d["d_min"] > d["d_max"]:
        print(f"ERROR: {d['n_layers']} layers cannot hold a 17-layer window", file=sys.stderr)
        return 1

    if a.sh:
        px = d["px_um"] if d["px_um"] else 8.64
        print(f"N_LAYERS={d['n_layers']}")
        print(f"PX_UM={px}")
        print(f"D_MIN={d['d_min']}")
        print(f"D_MAX={d['d_max']}")
        return 0

    px = d["px_um"]
    print(f"layers    {d['n_layers']}  (centre layer {d['mid']}, hecate zc at d=0: {d['hecate_zc0']:.2f})")
    if px:
        cm2 = (d["height"] * px / 1e4) * (d["width"] * px / 1e4)
        print(f"canvas    {d['height']} x {d['width']} px at {px} um = {cm2:.2f} cm2")
    else:
        print(f"canvas    {d['height']} x {d['width']} px (no pixel size in .zattrs)")
    print(f"depths    d = {d['d_min']}..{d['d_max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
