#!/usr/bin/env python3
"""Score an ensemble the same way on a known positive and on an unknown target.

The point is comparability. Run this on the PHerc0139 w035 control, where the answer is
known, then on a candidate patch, and read the candidate's numbers against the control's
rather than against intuition. Without the control row these numbers mean nothing.

Metrics, all restricted to pixels valid in every input:

  reading_vs_blind   Pearson r between the two faces of the sheet. Ink lives on one face,
                     so a true positive should be LOW. It is evidence about shared
                     artefact, not about ink strength.
  reading_vs_ct      r against the CT layer. High means the map is tracking density or
                     damage rather than ink.
  row_periodicity    strongest autocorrelation peak of the column-collapsed profile in the
                     3-12 mm band, where Herculaneum line spacing (5-9 mm) sits. A column
                     of text should show a clear positive peak; noise should not.
  dynamic_range      p99.5 - p1 of the ensemble over valid pixels.

With --labels (the control only) it also reports agreement with the hand-drawn ink
labels inside the supervision mask: ROC AUC, and the mean ensemble value on inked versus
uninked supervised pixels. AUC near 0.5 means the pipeline did not recover the ink that
is known to be there, which would indict the pipeline rather than the scroll.

    python sprint/score_maps.py --images /data/images/w035_control --name w035_control \\
        --labels /data/control/w035_control --px-um 9.362 --tag CONTROL
    python sprint/score_maps.py --images /data/images/20251028222030 \\
        --name 20251028222030 --px-um 8.64 --tag PHerc0800
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def load_gray(path: str) -> np.ndarray:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    return np.asarray(Image.open(path).convert("L")).astype(np.float32)


def pearson(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    x, y = a[m], b[m]
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den else float("nan")


def row_periodicity(img: np.ndarray, valid: np.ndarray, px_um: float) -> tuple[float, float]:
    """Strongest autocorrelation peak of the row profile, in the 3-12 mm band."""
    ys, xs = np.where(valid)
    if not len(ys):
        return float("nan"), float("nan")
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    h = min(900, (ys.max() - ys.min()) // 2, (xs.max() - xs.min()) // 2)
    if h < 100:
        return float("nan"), float("nan")
    blk = img[cy - h:cy + h, cx - h:cx + h].astype(np.float32).copy()
    blk -= blk.mean()
    prof = blk.mean(axis=1)                 # collapse along rows -> structure across rows
    prof -= prof.mean()
    ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
    if ac[0] == 0:
        return float("nan"), float("nan")
    ac = ac / ac[0]
    lags = np.arange(len(ac)) * px_um * 1e-3   # mm
    sel = (lags > 3) & (lags < 12)
    if not sel.any():
        return float("nan"), float("nan")
    i = int(np.argmax(ac[sel]))
    return float(ac[sel][i]), float(lags[sel][i])


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """AUC via rank statistic; no sklearn dependency."""
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks over ties
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="directory of ensemble_maps.py output")
    ap.add_argument("--name", required=True, help="map name prefix")
    ap.add_argument("--depth", default="d+00")
    ap.add_argument("--px-um", type=float, required=True)
    ap.add_argument("--labels", help="prefix of <p>__inklabels.npy / <p>__supervision.npy (control only)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--json-out")
    ap.add_argument("--all-stacks", action="store_true",
                    help="also score each model separately, not just the ensemble")
    a = ap.parse_args(argv)

    base = os.path.join(a.images, a.name)
    paths = {k: f"{base}__{k}__{a.depth}.png" for k in ("ensemble", "ensemble_blind", "ct")}
    missing = [k for k, p in paths.items() if not os.path.isfile(p)]
    if missing:
        print(f"missing: {[paths[k] for k in missing]}", file=sys.stderr)
        print("available:", sorted(os.path.basename(p) for p in glob.glob(base + "*.png"))[:8],
              file=sys.stderr)
        return 1

    ens, bli, ct = (load_gray(paths[k]) for k in ("ensemble", "ensemble_blind", "ct"))
    valid = (ens > 0) & (bli > 0) & (ct > 0)

    lo, hi = np.percentile(ens[valid], [1, 99.5])
    pk, pk_mm = row_periodicity(ens, valid, a.px_um)
    out = {
        "tag": a.tag or a.name, "name": a.name, "depth": a.depth, "px_um": a.px_um,
        "valid_px": int(valid.sum()),
        "valid_cm2": round(float(valid.sum()) * (a.px_um * 1e-4) ** 2, 3),
        "reading_vs_blind": round(pearson(ens, bli, valid), 3),
        "reading_vs_ct": round(pearson(ens, ct, valid), 3),
        "row_periodicity": round(pk, 3), "row_period_mm": round(pk_mm, 1),
        "dynamic_range": round(float(hi - lo), 1),
    }

    if a.labels:
        lab = np.load(a.labels + "__inklabels.npy")
        sup = np.load(a.labels + "__supervision.npy")
        if lab.shape != ens.shape:
            print(f"label shape {lab.shape} != image shape {ens.shape}; "
                  "the crop and the render must match", file=sys.stderr)
            return 1
        m = (sup > 0) & valid
        y = lab[m] > 0
        s = ens[m]
        out.update(supervised_px=int(m.sum()),
                   ink_frac_supervised=round(float(y.mean()), 4),
                   auc_vs_labels=round(roc_auc(s, y), 3),
                   mean_on_ink=round(float(s[y].mean()), 1) if y.any() else None,
                   mean_off_ink=round(float(s[~y].mean()), 1) if (~y).any() else None)

    if a.all_stacks:
        # Which models actually carry the signal? On the control this says whether a
        # slow model earns its runtime, and whether the ensemble beats its parts.
        rows = []
        for stack in ("ensemble", "dnative", "hecate", "ink9-42", "ink9-43"):
            p_read = f"{base}__{stack}__{a.depth}.png"
            p_blind = f"{base}__{stack}_blind__{a.depth}.png"
            if not os.path.isfile(p_read):
                continue
            r_img = load_gray(p_read)
            v = valid & (r_img > 0)
            row = {"stack": stack}
            if os.path.isfile(p_blind):
                b_img = load_gray(p_blind)
                row["r_blind"] = round(pearson(r_img, b_img, v & (b_img > 0)), 3)
            pk2, mm2 = row_periodicity(r_img, v, a.px_um)
            row["periodicity"] = round(pk2, 3)
            row["period_mm"] = round(mm2, 1)
            if a.labels:
                mm = (np.load(a.labels + "__supervision.npy") > 0) & v
                yy = np.load(a.labels + "__inklabels.npy")[mm] > 0
                row["auc"] = round(roc_auc(r_img[mm], yy), 3)
            rows.append(row)
        if rows:
            cols = [c for c in ("stack", "auc", "r_blind", "periodicity", "period_mm")
                    if any(c in r for r in rows)]
            w = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
            print(f"\n=== per stack at {a.depth} ===")
            print("  " + "  ".join(c.ljust(w[c]) for c in cols))
            for r in rows:
                print("  " + "  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))
            out["per_stack"] = rows

    width = max(len(k) for k in out)
    print(f"\n=== {out['tag']} ===")
    for k, v in out.items():
        if k not in ("tag", "per_stack"):
            print(f"  {k:<{width}}  {v}")

    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
