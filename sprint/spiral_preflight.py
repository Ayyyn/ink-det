#!/usr/bin/env python3
"""Check every input and tool the spiral chain needs, before the rental meter starts.

The chain's steps cost hours on a rented card. Most of the ways they fail are cheap to
detect first: a scroll whose lasagna prediction is missing, a `lasagna_scale` that does
not match the group actually being read, a pinned checkout whose scripts moved, a CT
volume id that is not the eligible one. This resolves all of that from S3 metadata and
the local checkout, prints what it found, and exits non-zero on anything that would
strand a run halfway.

Two checks here exist because they have already cost this project real time:

  * **`lasagna_scale` must match the real downsample factor of the chosen group FOR THAT
    SCROLL.** A mismatch silently reads wrong-resolution normals or throws an empty-z-ROI
    error. It is read from the store, never assumed.
  * **The eligible volume is pinned per scroll** in `prizeEligibility.json`. Fitting
    geometry against a different volume of the same scroll produces a perfectly good mesh
    that is not eligible for the prize. The volume id is checked against that list.

Script locations are *searched* rather than hardcoded: villa at the pinned commit
`0fb38c45` does not lay its spiral scripts out the way either a recent checkout or the
VC3D AppImage share directory does, so the chain resolves each tool by name and reports
where it landed.

    python sprint/spiral_preflight.py --scroll PHerc0268 --villa /data/villa --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BUCKET = "vesuvius-challenge-open-data"

# every executable step of the chain, by the name it is known by in villa / the winning repo
TOOLS = {
    "make_surf_sdt.py": "builds the capped signed-distance volume fit_spiral optimises against",
    "fit_spiral.py": "the spiral fit itself (headless; takes a dataset root)",
    "snap_meshes.py": "moves mesh nodes onto the predicted sheet - the winner's own contribution",
}


def s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config

    # The bucket is public and every call must be anonymous; a signed call fails outright.
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def ls_dirs(cli, prefix: str) -> list[str]:
    r = cli.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    return [p["Prefix"] for p in r.get("CommonPrefixes", [])]


def get_json(cli, key: str):
    return json.loads(cli.get_object(Bucket=BUCKET, Key=key)["Body"].read())


def eligible_volume(scroll: str, repo: Path) -> str | None:
    p = repo / "villa" / "scrollprize.org" / "src" / "data" / "prizeEligibility.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    for e in d.get("first-letters-2027", []):
        if e.get("scroll") == scroll:
            return e.get("volume")
    return None


def find_tool(villa: Path, name: str) -> str | None:
    hits = sorted(villa.rglob(name))
    return str(hits[0]) if hits else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--villa", default=os.environ.get("VILLA", ""), help="pinned villa checkout")
    ap.add_argument("--vc3d", default=os.environ.get("VC3D", ""), help="VC3D AppImage (for the render)")
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    a = ap.parse_args(argv)

    repo = Path(a.repo)
    out: dict = {"scroll": a.scroll}
    problems: list[str] = []
    cli = s3()

    # --- the eligible volume -------------------------------------------------------
    want = eligible_volume(a.scroll, repo)
    out["eligible_volume"] = want
    print(f"=== {a.scroll}")
    print(f"  eligible volume      {want or 'UNKNOWN (prizeEligibility.json not found)'}")

    vols = ls_dirs(cli, f"{a.scroll}/volumes/")
    vol = next((v for v in vols if want and want in v), None)
    if want and not vol:
        problems.append(f"eligible volume {want} has no store under {a.scroll}/volumes/")
    for v in vols:
        name = v.rstrip("/").split("/")[-1]
        print(f"  volume               {name}{'   <- eligible' if vol == v else ''}")
    out["volume_store"] = f"s3://{BUCKET}/{vol}" if vol else None

    # --- surface prediction (m7) ---------------------------------------------------
    surfs = [s for s in ls_dirs(cli, f"{a.scroll}/representations/predictions/surfaces/")
             if s.rstrip("/").endswith(".zarr")]
    surf = next((s for s in surfs if want and want in s), None)
    out["surface"] = f"s3://{BUCKET}/{surf}" if surf else None
    if not surf:
        problems.append("no m7 surface prediction on the eligible volume - the chain cannot start")
    print(f"  surface m7           {surf.rstrip('/').split('/')[-1] if surf else 'MISSING'}")

    # --- lasagna, and the scale that must be handed to fit_spiral ------------------
    las_root = ls_dirs(cli, f"{a.scroll}/representations/predictions/lasagna/")
    las = next((l for l in las_root if want and want in l), las_root[0] if las_root else None)
    out["lasagna"] = f"s3://{BUCKET}/{las}" if las else None
    if not las:
        problems.append("no lasagna prediction - fit_spiral has no normals; grow lasagna first")
        print("  lasagna              MISSING")
    else:
        print(f"  lasagna              {las.rstrip('/').split('/')[-1]}")
        comps = {c.rstrip("/").split("/")[-1] for c in ls_dirs(cli, las)}
        need = {f"{a.scroll}_{k}.ome.zarr" for k in ("nx", "ny", "grad_mag")}
        missing = need - comps
        if missing:
            problems.append(f"lasagna is missing {sorted(missing)}")
        try:
            attrs = get_json(cli, f"{las}{a.scroll}_nx.ome.zarr/.zattrs")
            table = {d["path"]: float(d["coordinateTransformations"][0]["scale"][0])
                     for d in attrs["multiscales"][0]["datasets"]}
            out["lasagna_groups"] = table
            print(f"  lasagna groups       {table}")
            # the winning run used group 2; whatever group is used, this is its scale
            g = "2" if "2" in table else sorted(table, key=lambda k: table[k])[0]
            out["lasagna_group"], out["lasagna_scale"] = g, table[g]
            print(f"  -> use group {g} with lasagna_scale {table[g]:g}  "
                  f"(READ from the store; a mismatch here fails silently)")
        except Exception as e:
            problems.append(f"could not read lasagna scales: {type(e).__name__}: {e}")

    # --- is the geometry already done for us? --------------------------------------
    segs = ls_dirs(cli, f"{a.scroll}/segments/")
    out["published_segments"] = len(segs)
    print(f"  published segments   {len(segs)}"
          + ("  <- geometry already exists; check these before fitting a spiral" if segs else ""))

    # --- local tools ---------------------------------------------------------------
    print()
    if a.villa:
        villa = Path(a.villa)
        if not villa.is_dir():
            problems.append(f"villa checkout not found: {villa}")
        else:
            found = {}
            for name, why in TOOLS.items():
                p = find_tool(villa, name)
                found[name] = p
                print(f"  {name:20} {p or 'NOT FOUND'}")
                if not p:
                    problems.append(f"{name} not in the checkout ({why})")
            out["tools"] = found
    else:
        print("  (no --villa given; tool locations unchecked)")

    if a.vc3d:
        ok = Path(a.vc3d).exists()
        out["vc3d"] = a.vc3d
        print(f"  {'VC3D':20} {a.vc3d}{'' if ok else '   NOT FOUND'}")
        if not ok:
            problems.append(f"VC3D AppImage not found: {a.vc3d}")

    print()
    if problems:
        print(f"{len(problems)} problem(s) - do not start the chain:")
        for p in problems:
            print(f"  * {p}")
    else:
        print("preflight clean.")
    out["problems"] = problems

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {a.json_out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
