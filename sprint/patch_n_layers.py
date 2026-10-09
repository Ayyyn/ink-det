#!/usr/bin/env python3
"""Make the PHerc0343 ensemble step work on renders that are not 66 layers deep.

`ensemble_maps.py` assumes a 66-layer render in four places: the two map-filename
f-strings, the `N66, MID = 66, 33` constants, and an assert on the surface volume's
depth. The published surface volumes of PHerc0800 are 31 layers, so the sweep writes
L31s<s> / L31zc<zc> names and the assert fires before anything is read.

Every substitution derives from N_LAYERS (default 66), and at N=66 each one is
character-for-character equivalent to the original:

    MID            = 66 // 2      = 33                    (unchanged)
    villa map name = 66 // 2 + d - 8 = d + 25             (unchanged)
    hecate zc      = (66 - 1) / 2 + d = 32.50 + d         (unchanged)
    CT stat slice  = [33-15 : 33+16] = [18:49]            (unchanged; clamped for N < 31)

so Erwin Nieuwlaar's own reproduction is unaffected by applying this.

Idempotent, and it upgrades a file patched by an earlier version of this script.

    python sprint/patch_n_layers.py /path/to/pherc343/ink/scripts/ensemble_maps.py
"""
import re
import sys
from pathlib import Path

DECL = "_N = int(os.environ.get('N_LAYERS', '66'))  # patched: see sprint/patch_n_layers.py"

# (description, old, new) - each applied only if `old` is still present
SUBS = [
    ("hecate map name",
     "f'{name}__hecate__L66zc{32.5 + d:.2f}{rev}.png'",
     "f'{name}__hecate__L{_N}zc{(_N - 1) / 2 + d:.2f}{rev}.png'"),
    ("villa map name",
     "f'{name}__{mk}__L66s{d + 25}{rev}.tif'",
     "f'{name}__{mk}__L{_N}s{_N // 2 + d - 8}{rev}.tif'"),
    ("layer-count constants",
     "N66, MID = 66, 33",
     "N66, MID = _N, _N // 2"),
    ("CT statistics slice",
     "sel = blk[MID - 15:MID + 16]",
     "sel = blk[max(0, MID - 15):min(N66, MID + 16)]"),
    ("CT statistics metadata",
     "layers=[MID - 15, MID + 15]",
     "layers=[max(0, MID - 15), min(N66 - 1, MID + 15)]"),
]


def main(path: str) -> int:
    p = Path(path)
    if not p.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 1
    src = p.read_text(encoding="utf-8")
    before = src

    if DECL not in src:
        src = re.sub(r"^(import argparse.*)$", r"\1\n" + DECL, src, count=1, flags=re.M)
        if DECL not in src:
            print("ERROR: could not place the N_LAYERS declaration", file=sys.stderr)
            return 1

    applied, already = [], []
    for what, old, new in SUBS:
        if old in src:
            src = src.replace(old, new)
            applied.append(what)
        elif new in src:
            already.append(what)
        else:
            print(f"ERROR: neither old nor new form found for {what!r}; upstream changed",
                  file=sys.stderr)
            return 1

    if src != before:
        p.write_text(src, encoding="utf-8")
    print(f"{p}: patched [{', '.join(applied) or 'nothing'}]"
          f"{'; already done: ' + ', '.join(already) if already else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
