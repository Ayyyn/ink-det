#!/usr/bin/env python3
"""Copy one published surface volume (render) from S3 to a local directory.

The ink sweep wants the render on local disk: villa scans it for occupied patches
at full resolution and reads it many times over, once per model per depth. These
are uint8 and uncompressed, so a 2580x2580x31 render is ~200 MB and lands in well
under a minute.

    python sprint/pull_render.py --scroll PHerc0800 --segment 20251028222030 --out /data
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
for _n in ("s3fs", "aiobotocore", "botocore"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

BUCKET = "vesuvius-challenge-open-data"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--segment", required=True, help="substring of the segment name")
    ap.add_argument("--out", required=True, help="local directory to write <segment>.zarr into")
    ap.add_argument("--levels", default="0", help="comma-separated pyramid levels to copy, or 'all' (default 0)")
    args = ap.parse_args(argv)

    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    base = f"{BUCKET}/{args.scroll}/segments"
    segs = [p for p in fs.ls(base) if args.segment in p.split("/")[-1]]
    if len(segs) != 1:
        print(f"need exactly one match, got {[s.split('/')[-1] for s in segs]}", file=sys.stderr)
        return 1
    seg = segs[0]
    tag = seg.split("/")[-1].split("-")[0]
    zarrs = [q for q in fs.ls(f"{seg}/surface-volumes") if q.endswith(".zarr")]
    if not zarrs:
        print("no surface volume published for this segment", file=sys.stderr)
        return 1
    src = zarrs[0]

    dst = Path(args.out) / f"{tag}.zarr"
    dst.mkdir(parents=True, exist_ok=True)
    for meta in (".zgroup", ".zattrs"):
        if fs.exists(f"{src}/{meta}"):
            (dst / meta).write_bytes(fs.cat(f"{src}/{meta}"))

    levels = [p.split("/")[-1] for p in fs.ls(src) if p.split("/")[-1].isdigit()]
    if args.levels != "all":
        want = set(args.levels.split(","))
        levels = [lv for lv in levels if lv in want]
    for lv in sorted(levels):
        t0 = time.time()
        fs.get(f"{src}/{lv}", str(dst / lv), recursive=True)
        n = sum(1 for _ in (dst / lv).rglob("*") if _.is_file())
        print(f"level {lv}: {n} files in {time.time() - t0:.0f}s", flush=True)
    print(f"\n{dst}")
    print("note: run_ink.sh presents a multi-level render as level 0 only via a symlink;")
    print("copying level 0 alone (the default) gives the same thing with less traffic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
