#!/usr/bin/env python3
"""Make the PHerc0343 ensemble step work on renders that are not 66 layers deep.

ensemble_maps.py hardcodes the 66-layer map naming in two f-strings. The published
surface volumes of PHerc0800 are 31 layers, so the same sweep writes L31s<s> /
L31zc<zc> and the ensemble finds nothing. This rewrites both to read N_LAYERS from
the environment, defaulting to 66 — for N=66 the expressions are identical to the
originals ((66-1)/2 = 32.5 and 66//2 + d - 8 = d + 25), so his reproduction is
untouched.

    python sprint/patch_n_layers.py /path/to/pherc343/ink/scripts/ensemble_maps.py
"""
import re
import sys
from pathlib import Path

HEC_OLD = "f'{name}__hecate__L66zc{32.5 + d:.2f}{rev}.png'"
HEC_NEW = "f'{name}__hecate__L{_N}zc{(_N - 1) / 2 + d:.2f}{rev}.png'"
VIL_OLD = "f'{name}__{mk}__L66s{d + 25}{rev}.tif'"
VIL_NEW = "f'{name}__{mk}__L{_N}s{_N // 2 + d - 8}{rev}.tif'"
DECL = "_N = int(os.environ.get('N_LAYERS', '66'))  # patched: see sprint/patch_n_layers.py"


def main(path: str) -> int:
    p = Path(path)
    src = p.read_text(encoding="utf-8")
    if DECL in src:
        print("already patched")
        return 0
    missing = [o for o in (HEC_OLD, VIL_OLD) if o not in src]
    if missing:
        print("ERROR: expected strings not found; upstream changed:", missing, file=sys.stderr)
        return 1
    src = src.replace(HEC_OLD, HEC_NEW).replace(VIL_OLD, VIL_NEW)
    src = re.sub(r"^(import argparse.*)$", r"\1\n" + DECL, src, count=1, flags=re.M)
    if DECL not in src:
        print("ERROR: could not place the N_LAYERS declaration", file=sys.stderr)
        return 1
    p.write_text(src, encoding="utf-8")
    print(f"patched {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
