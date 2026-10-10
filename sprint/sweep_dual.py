#!/usr/bin/env python3
"""Run the ink sweep across every available GPU, with live progress.

Erwin Nieuwlaar's run_depth_sweep.sh starts its villa and hecate queues inside one
process, so both inherit one CUDA_VISIBLE_DEVICES and a second GPU sits idle. Splitting
the model list across two processes fixes that without touching his script:

    GPU 0   dnative, ink_9um seed42, ink_9um seed43     (fast: ~35 s per job)
    GPU 1   hecate                                      (slow: minutes per job)

That pairing is deliberate. On the first PHerc0800 run the three villa models finished
all 21 jobs in about the time two hecate jobs took, so hecate alone sets the wall-clock.
Giving it a dedicated card and its own concurrency roughly halves the sweep.

With one GPU it runs a single process with both queues, exactly as before.

Progress is printed as a heartbeat with a per-model breakdown. Maps appear only when a
job finishes - each writes to a temp dir and moves at the end - so a flat counter
between completions is normal, not a stall. The sweep is resumable: finished maps are
skipped, and depths run from d = 0 outwards.

    python sprint/sweep_dual.py --render /data/control/w035_control.zarr \\
        --out /data/maps/w035 --dmin -3 --dmax 3
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path

VILLA = ["dnative", "hybrid_3d2d-seed42", "hybrid_3d2d-seed43"]
HECATE = ["hecate"]


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        return len([l for l in out.splitlines() if l.strip().startswith("GPU ")])
    except Exception:
        return 0


def count_maps(out: Path) -> dict[str, int]:
    seen: dict[str, int] = {}
    for p in glob.glob(str(out / "*.tif")) + glob.glob(str(out / "*.png")):
        m = re.search(r"__([A-Za-z0-9_.-]+?)__L\d+", os.path.basename(p))
        if m:
            seen[m.group(1)] = seen.get(m.group(1), 0) + 1
    return seen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dmin", type=int, default=-3)
    ap.add_argument("--dmax", type=int, default=3)
    ap.add_argument("--villa-par", type=int, default=2, help="concurrent villa jobs (~3 GB each)")
    ap.add_argument("--hec-par", type=int, default=2, help="concurrent hecate jobs")
    ap.add_argument("--gpus", default=None, help="comma-separated ids; default: all visible")
    ap.add_argument("--poll", type=int, default=60, help="heartbeat seconds")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit; checks the launch path without a GPU")
    ap.add_argument("--models", default=None,
                    help="space-separated model list. Default drops hecate: on the "
                         "PHerc0139 control the three villa models ensemble to AUC 0.979 "
                         "against 0.972 with hecate included, and hecate costs ~14x more "
                         "per job. Pass all four for a final candidate image.")
    a = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    sweep = here / "sweep.sh"
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    if a.gpus:
        gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    else:
        gpus = [str(i) for i in range(gpu_count())]
    n_depths = a.dmax - a.dmin + 1

    models = a.models.split() if a.models else list(VILLA)
    has_hec = "hecate" in models
    # plan entries: (label, models, gpu, villa_par, hec_par, dmin, dmax)
    if len(gpus) >= 2 and has_hec and len(models) > 1:
        # hecate is ~14x slower per job, so it gets a card to itself
        plan = [("villa", [m for m in models if m != "hecate"], gpus[0], a.villa_par, 1, a.dmin, a.dmax),
                ("hecate", ["hecate"], gpus[1], 1, a.hec_par, a.dmin, a.dmax)]
        print(f"2+ GPUs: villa models on GPU {gpus[0]}, hecate on GPU {gpus[1]}")
    elif len(gpus) >= 2:
        # comparable per-job cost, so split the depth range instead
        mid = (a.dmin + a.dmax) // 2
        plan = [("lower", models, gpus[0], a.villa_par, a.hec_par, a.dmin, mid),
                ("upper", models, gpus[1], a.villa_par, a.hec_par, mid + 1, a.dmax)]
        print(f"2+ GPUs: d {a.dmin}..{mid} on GPU {gpus[0]}, d {mid + 1}..{a.dmax} on GPU {gpus[1]}")
    else:
        plan = [("all", models, gpus[0] if gpus else None, a.villa_par, a.hec_par, a.dmin, a.dmax)]
        print(f"single GPU{' ' + gpus[0] if gpus else ''}: one queue")

    total = len(models) * n_depths * 2      # both faces per job
    if a.dry_run:
        print(f"\nexpecting {total} maps ({len(models)} models x {n_depths} depths x 2 faces)")
        for name, mods, gpu, vp, hp, dlo, dhi in plan:
            print(f"  {name:7} GPU {gpu}  d {dlo}..{dhi}  villa_par {vp} hec_par {hp}  "
                  f"models: {' '.join(mods)}")
        return 0

    procs = []
    for name, mods, gpu, vp, hp, dlo, dhi in plan:
        env = dict(os.environ)
        env["MODELS"] = " ".join(mods)
        env["P_VILLA"], env["P_HEC"] = str(vp), str(hp)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        log = open(out / "logs" / f"queue_{name}.log", "w")
        p = subprocess.Popen(["bash", str(sweep), a.render, str(out), str(dlo), str(dhi)],
                             env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((name, p, log))
        print(f"  started {name:7} pid {p.pid}  d {dlo}..{dhi}  models: {' '.join(mods)}")

    print(f"\nexpecting {total} maps ({len(models)} models x {n_depths} depths x 2 faces)\n",
          flush=True)
    t0 = time.time()
    try:
        while any(p.poll() is None for _, p, _ in procs):
            seen = count_maps(out)
            n = sum(seen.values())
            brk = "  ".join(f"{k.replace('hybrid_3d2d-', '')}:{v}" for k, v in sorted(seen.items()))
            print(f"{int(time.time() - t0) // 60:4d} min   {n:3d}/{total} maps   {brk}", flush=True)
            time.sleep(a.poll)
    except KeyboardInterrupt:
        print("\ninterrupted; terminating queues (finished maps are kept and will be skipped on resume)")
        for _, p, _ in procs:
            p.terminate()
        raise

    rc = 0
    for name, p, log in procs:
        log.close()
        if p.returncode != 0:
            rc = p.returncode
            print(f"queue {name} exited {p.returncode}; tail of its log:")
            print("".join(open(out / "logs" / f"queue_{name}.log", encoding="utf-8",
                               errors="replace").readlines()[-15:]))

    seen = count_maps(out)
    print(f"\ndone in {(time.time() - t0) / 60:.0f} min: {sum(seen.values())}/{total} maps")
    for k, v in sorted(seen.items()):
        print(f"   {k:24} {v}")

    tim = out / "logs" / "timings.tsv"
    if tim.exists():
        rows = [l.split("\t") for l in tim.read_text(encoding="utf-8").splitlines() if l.count("\t") == 3]
        if rows:
            per: dict[str, list[int]] = {}
            for model, _d, secs, _rc in rows:
                per.setdefault(model, []).append(int(secs))
            print("\nseconds per job:")
            for k, v in sorted(per.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
                print(f"   {k:24} median {sorted(v)[len(v) // 2]:5d}   n={len(v)}")
            bad = [r for r in rows if r[3].strip() != "0"]
            if bad:
                print(f"\n{len(bad)} failed job(s): {[(r[0], r[1]) for r in bad][:10]}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
