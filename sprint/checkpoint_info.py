#!/usr/bin/env python3
"""Print what each ink checkpoint actually expects: input depth, preprocessing, patch size.

This decides whether the sweep is aimed correctly. villa's flat inference takes the
half-open layer range given by --layer-start/--layer-end, centre-crops it to the
checkpoint's own `input_depth`, and - if the selection is SHORTER than input_depth -
centre-pads it with zeros (models/input_padding.py). Padding is silent. A model asked
for 21 layers but handed 17 still runs, and still produces a plausible-looking map,
with two layers of zeros glued to each face.

Erwin Nieuwlaar's run_ink.sh passes exactly 17 layers (s..s+16). That is right only if
input_depth <= 17 for every model. Worth confirming rather than assuming, because the
failure is invisible.

    python sprint/checkpoint_info.py /data/ckpt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys


def describe(path: str) -> dict:
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = {"file": os.path.basename(path)}
    if not isinstance(ck, dict):
        out["note"] = f"not a dict checkpoint ({type(ck).__name__})"
        return out

    out["keys"] = ",".join(sorted(k for k in ck if not k.endswith("state_dict")))[:120]
    cfg = None
    for key in ("config", "cfg", "train_config", "hyper_parameters"):
        if isinstance(ck.get(key), dict):
            cfg = ck[key]
            out["config_key"] = key
            break
    if cfg is None:
        out["note"] = "no embedded config found"
        return out

    def dig(d, *names):
        """Find the first matching key at any nesting depth."""
        stack = [d]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                for k, v in cur.items():
                    if k in names:
                        return v
                    if isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(cur, list):
                stack.extend(x for x in cur if isinstance(x, (dict, list)))
        return None

    for label, names in (
        ("input_depth", ("input_depth",)),
        ("input_pad_depth_to", ("input_pad_depth_to",)),
        ("mode", ("mode",)),
        ("patch_size", ("patch_size", "tile_size", "crop_size")),
        ("normalization", ("normalization", "normalize", "preprocessing")),
        ("window_depth", ("window_depth",)),
        ("label_smoothing", ("label_smoothing",)),
        ("wandb_run", ("wandb_run_id", "run_id")),
        ("step", ("step", "global_step")),
    ):
        v = dig(cfg, *names) if label != "step" else (ck.get("step") or dig(cfg, *names))
        if v is not None:
            out[label] = v if not isinstance(v, (dict, list)) else str(v)[:80]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_dir")
    a = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(a.ckpt_dir, "*.pth")))
    if not files:
        print(f"no .pth under {a.ckpt_dir}", file=sys.stderr)
        return 1

    depths = {}
    for f in files:
        try:
            d = describe(f)
        except Exception as e:
            d = {"file": os.path.basename(f), "note": f"{type(e).__name__}: {e}"}
        print(f"\n=== {d.pop('file')}")
        for k, v in d.items():
            print(f"   {k:20} {v}")
        if "input_depth" in d:
            depths[os.path.basename(f)] = int(d["input_depth"])

    if depths:
        worst = max(depths.values())
        print(f"\ndeepest model wants {worst} layers; run_ink.sh supplies 17")
        if worst > 17:
            print("  *** MISMATCH: the selection is shorter than the model wants, so villa")
            print("      centre-pads it with ZEROS. Every map so far is degraded. Widen the")
            print("      window in run_ink.sh to at least this many layers and re-run.")
        else:
            print("  OK: 17 >= every model's input depth, so the centre crop is exact "
                  "and nothing is zero-padded.")
        n_needed = worst + 2 * 7
        print(f"\nA sweep of d = -7..+7 therefore touches {n_needed} source layers at most.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
