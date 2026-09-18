#!/usr/bin/env python3
"""Cross-reference the First Letters / Grand Prize eligible volume list against
the data-browser index, and report what precomputed assets exist per target.

Eligibility pins ONE volume id per scroll. Anything published against a different
volume of the same scroll is not usable for a First Letters submission.

    python eligible_assets.py --villa ../villa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--villa", type=Path, default=Path(__file__).resolve().parents[1] / "villa")
    ap.add_argument("--prize", default="first-letters-2027",
                    help="key in prizeEligibility.json (first-letters-2027 | grand-prize-2027)")
    args = ap.parse_args()

    elig_path = args.villa / "scrollprize.org/src/data/prizeEligibility.json"
    index_path = args.villa / "scrollprize.org/static/data_browser/index.json"

    elig = json.loads(elig_path.read_text(encoding="utf-8"))[args.prize]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in index["scrolls"]}

    print(f"data-browser index updated: {index.get('updated')}")
    print(f"eligible scrolls: {len(elig)}\n")

    hdr = f"{'scroll':<12} {'volume':<15} {'px_um':>6} {'segs':>5} {'surf':>5} {'lasa':>5} {'ink3d':>6} {'score':>6}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for entry in elig:
        sid, vol = entry["scroll"], entry["volume"]
        s = by_id.get(sid)
        if s is None:
            print(f"{sid:<12} {vol:<15}  !! not in data-browser index")
            continue

        # resolution of the ELIGIBLE volume specifically, not the scroll minimum
        px = next((v["px"] for v in s.get("ctVolumes", []) if v["id"] == vol), None)

        preds = [p for p in (s.get("predictions") or []) if p.get("baseVolume") == vol]
        surf = any(p["purpose"] == "surface-prediction" for p in preds)
        lasa = any(p["purpose"] == "lasagna" for p in preds)

        rows.append({
            "scroll": sid, "volume": vol, "px": px,
            "segments": s.get("n_segments", 0),
            "surface": surf, "lasagna": lasa,
            "ink3d": bool(s.get("hasInk3d")),
            "score": (s.get("progress") or {}).get("score"),
            "stages": s.get("stages", {}),
        })

        print(f"{sid:<12} {vol:<15} {px if px is not None else '?':>6} "
              f"{s.get('n_segments', 0):>5} {'yes' if surf else '-':>5} "
              f"{'yes' if lasa else '-':>5} {'yes' if s.get('hasInk3d') else '-':>6} "
              f"{(s.get('progress') or {}).get('score', '?'):>6}")

    print()
    px_vals = sorted({r["px"] for r in rows if r["px"]})
    print(f"resolutions present among eligible volumes: {px_vals}")
    print(f"with surface prediction : {sum(r['surface'] for r in rows)}/{len(rows)}")
    print(f"with lasagna prediction : {sum(r['lasagna'] for r in rows)}/{len(rows)}")
    print(f"with existing segments  : {sum(r['segments'] > 0 for r in rows)}/{len(rows)}")
    already = [r["scroll"] for r in rows if r["stages"].get("text")]
    print(f"already read (text found): {already or 'none'}")

    out = Path(__file__).resolve().parent / f"eligible_assets_{args.prize}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
