#!/usr/bin/env python3
"""Does PHerc1203's 2.4 um scan cover the whole roll, or only a slab?

This is the fact that decides your primary target. Costs nothing to answer:
it reads Zarr metadata over anonymous HTTP, a few KB, no GPU, no rental.

    pip install s3fs zarr
    python check_1203.py

If the 2.403 um volume spans most of the scroll's height, PHerc1203 is target #1
and you rent for 2.4 um work. If it is a thin slab, the 2.4 um play is limited to
that band and PHerc1447 becomes the primary.
"""

from __future__ import annotations

import json
import sys

BUCKET = "vesuvius-challenge-open-data"
VOLUMES = {
    "PHerc1203 @ 2.403um (77keV)": (
        f"{BUCKET}/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr",
        2.403,
    ),
    "PHerc1203 @ 9.362um (113keV)": (
        f"{BUCKET}/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr",
        9.362,
    ),
    "PHerc1447 @ 8.640um (116keV)": (
        f"{BUCKET}/PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr",
        8.640,
    ),
}


def main() -> int:
    try:
        import s3fs
    except ImportError:
        print("pip install s3fs zarr", file=sys.stderr)
        return 1

    fs = s3fs.S3FileSystem(anon=True)

    for label, (path, px_um) in VOLUMES.items():
        print(f"\n=== {label} ===")
        print(f"    {path}")
        try:
            entries = sorted(e.rsplit("/", 1)[-1] for e in fs.ls(path))
        except Exception as exc:  # noqa: BLE001
            print(f"    !! cannot list: {exc}")
            continue
        print(f"    contents: {', '.join(entries[:12])}")

        # pyramid level 0 metadata, zarr v2 or v3
        meta = None
        for candidate in (f"{path}/0/.zarray", f"{path}/0/zarr.json", f"{path}/.zarray"):
            try:
                with fs.open(candidate, "rb") as fh:
                    meta = json.load(fh)
                print(f"    metadata: {candidate.rsplit('/', 2)[-2:]}")
                break
            except Exception:  # noqa: BLE001, S110
                continue

        if meta is None:
            print("    !! no level-0 array metadata found — inspect contents by hand")
            continue

        shape = meta.get("shape")
        chunks = meta.get("chunks") or (meta.get("chunk_grid", {})
                                        .get("configuration", {}).get("chunk_shape"))
        dtype = meta.get("dtype") or meta.get("data_type")

        print(f"    shape  : {shape}   dtype {dtype}   chunks {chunks}")

        if shape and len(shape) == 3:
            z, y, x = shape
            mm = [n * px_um / 1000.0 for n in (z, y, x)]
            print(f"    physical: Z {mm[0]:.1f} mm   Y {mm[1]:.1f} mm   X {mm[2]:.1f} mm")
            gb = z * y * x / 1e9
            print(f"    voxels : {gb:.1f} G  (~{gb:.0f} GB uncompressed at 1 byte/voxel)")
            if mm[0] < 40:
                print("    >> Z span is short — this looks like a SLAB, not the full roll.")
            else:
                print("    >> Z span is substantial — plausibly full-height coverage.")

    print(
        "\nRead: a Herculaneum roll is roughly 50-60 mm across and typically "
        "100-250 mm tall.\nCompare the 2.403 um Z span against the 9.362 um one for "
        "the same scroll — that\nratio is the honest answer to 'whole roll or slab'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
