#!/usr/bin/env python3
"""Push, run, watch and pull the Kaggle notebook from here, so cells stop being retyped.

Every invocation error so far has been transport, not science: a shell command pasted
into a Python cell, a backslash continuation the `!` magic ignores, a stale cell left
over from hand-editing. The Kaggle API removes the retyping step entirely - the notebook
is generated from `_make_notebook.py`, validated by `_check_notebook.py`, pushed, run
headless, and its results pulled back - so the file in the repo is always what ran.

    python sprint/kaggle_sync.py push --what pherc0800 --seg 20251028222030 --dmin -7 --dmax 7
    python sprint/kaggle_sync.py status
    python sprint/kaggle_sync.py pull --out private/kaggle_pull   # what is on Kaggle now
    python sprint/kaggle_sync.py output --out private/my_data/kaggle

Three things worth knowing before using it:

  * **A push is a run.** The API has no save-without-execute: `kernels push` commits a new
    version and immediately queues it. Each one bills against the ~30 GPU-h weekly quota,
    so do not push to "check" something - that is what `--dry-run` and the validator are
    for.
  * **A pushed run is headless**, which suits a 40-minute sweep better than babysitting a
    browser: it survives a closed laptop, and `status` plus `output` read it back. Nothing
    is visible until it finishes or fails, so the notebook must save its own evidence -
    it does, into `/kaggle/working`, which is what `output` downloads.
  * **`enable_gpu` alone does not pin the accelerator.** `--accelerator NvidiaTeslaT4` is
    passed for us, but confirm in the run log that cell 1 printed *two* GPUs: with one,
    `sweep_dual.py` falls back to a single queue and the sweep takes twice as long.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NB = HERE / "kaggle_sweep.ipynb"
SLUG = "vesuvius-ink-sweep"
TITLE = "Vesuvius ink sweep"


def kaggle_cmd() -> list[str]:
    """The kaggle CLI, preferring the repo venv that already has it."""
    for cand in (REPO / ".venv-check" / "Scripts" / "kaggle.exe",
                 REPO / ".venv-check" / "Scripts" / "kaggle",
                 REPO / ".venv-check" / "bin" / "kaggle"):
        if cand.exists():
            return [str(cand)]
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    return [sys.executable, "-m", "kaggle"]


KDIR = Path.home() / ".kaggle"
TOKEN_FILES = (KDIR / "access_token", KDIR / "access_token.txt")

AUTH_HELP = f"""no Kaggle credentials found. Two ways, either is fine:

  1. OAuth, nothing to copy (recommended by the CLI itself):
         python sprint/kaggle_sync.py auth --login
  2. A token string from kaggle.com/settings -> API -> Create New Token.
     Newer Kaggle shows the value to COPY instead of downloading kaggle.json:
         python sprint/kaggle_sync.py auth --token <paste-it-here>
     which writes {KDIR / 'access_token'} without the BOM Notepad would add."""


def have_token() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY"):
        return True
    return (any(f.exists() and f.read_text(encoding="utf-8-sig").strip() for f in TOKEN_FILES)
            or (KDIR / "credentials.json").exists()
            or (KDIR / "kaggle.json").exists())


def username(required: bool = True) -> str:
    """Resolve the account name. A bare access_token carries no username, so ask the API."""
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    for f, key in ((KDIR / "credentials.json", "username"), (KDIR / "kaggle.json", "username"),
                   (KDIR / "sync_user.json", "username")):
        try:
            v = json.loads(f.read_text(encoding="utf-8-sig")).get(key)
            if v:
                return v
        except Exception:
            continue
    if have_token():
        # `kernels list --mine` returns refs shaped `username/slug`; cheapest whoami the
        # CLI offers. Cached, because it costs a round trip.
        r = subprocess.run(kaggle_cmd() + ["kernels", "list", "--mine", "--page-size", "1", "-v"],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "/" in line and not line.lower().startswith("ref"):
                name = line.split(",")[0].split("/")[0].strip()
                if name:
                    KDIR.mkdir(parents=True, exist_ok=True)
                    (KDIR / "sync_user.json").write_text(
                        json.dumps({"username": name}), encoding="utf-8")
                    return name
    if not required:
        return "<your-kaggle-username>"
    sys.exit(AUTH_HELP + "\n\nOr pass it explicitly: --user <name>")


def cmd_auth(a) -> int:
    KDIR.mkdir(parents=True, exist_ok=True)
    if a.login:
        rc = run(kaggle_cmd() + ["auth", "login"]).returncode
        if rc != 0:
            return rc
    elif a.token:
        # utf-8 with no BOM and no trailing newline: the reader strips whitespace but a
        # BOM becomes part of the bearer token and every call 401s with no clue why.
        tok = a.token.strip().strip('"').strip("'")
        (KDIR / "access_token").write_text(tok, encoding="utf-8")
        for stale in (KDIR / "access_token.txt",):
            if stale.exists():
                stale.unlink()      # it would never be read, but it would confuse the next reader
        print(f"wrote {KDIR / 'access_token'}  ({len(tok)} chars)")
    elif not have_token():
        print(AUTH_HELP)
        return 1

    (KDIR / "sync_user.json").unlink(missing_ok=True)
    who = username()
    print(f"authenticated as {who}")
    return run(kaggle_cmd() + ["quota"]).returncode


# --- config injection ------------------------------------------------------------------
# The notebook keeps every knob in one cell so a run is described by five lines. Setting
# them from here means a run can be launched without opening a browser, and the committed
# notebook records exactly what was run.
KNOBS = {
    "what": ("WHAT", repr),
    "seg": ("SEG", repr),
    "dmin": ("DMIN", str),
    "dmax": ("DMAX", str),
    "models": ("MODELS", repr),
    "villa_par": ("VILLA_PAR", str),
    "hec_par": ("HEC_PAR", str),
}


def set_knob(src: str, name: str, value: str) -> tuple[str, bool]:
    """Rewrite `NAME = ...`, including the paired `DMIN, DMAX = a, b` form."""
    pair = {"DMIN": ("DMIN, DMAX", 0), "DMAX": ("DMIN, DMAX", 1),
            "VILLA_PAR": ("VILLA_PAR, HEC_PAR", 0), "HEC_PAR": ("VILLA_PAR, HEC_PAR", 1)}
    if name in pair:
        lhs, idx = pair[name]
        m = re.search(rf"^{re.escape(lhs)} = (.+?)(\s*#.*)?$", src, re.M)
        if not m:
            return src, False
        parts = [p.strip() for p in m.group(1).split(",")]
        parts[idx] = value
        return src[:m.start(1)] + ", ".join(parts) + src[m.end(1):], True
    m = re.search(rf"^{re.escape(name)}(\s*)= (.+?)(\s*#.*)?$", src, re.M)
    if not m:
        return src, False
    return src[:m.start(2)] + value + src[m.end(2):], True


def apply_config(nb_path: Path, overrides: dict) -> dict:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cell = next((c for c in nb["cells"]
                 if c["cell_type"] == "code" and "".join(c["source"]).startswith("WHAT")), None)
    if cell is None:
        sys.exit("no config cell found (expected a code cell starting with WHAT =)")
    src = "".join(cell["source"])
    applied = {}
    for key, value in overrides.items():
        if value is None:
            continue
        name, fmt = KNOBS[key]
        src, ok = set_knob(src, name, fmt(value))
        if not ok:
            sys.exit(f"could not find {name} in the config cell")
        applied[name] = value
    cell["source"] = [l + "\n" for l in src.split("\n")]
    if cell["source"] and cell["source"][-1] == "\n":
        cell["source"].pop()
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return applied


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("$ kaggle", " ".join(map(str, cmd[1:])))
    return subprocess.run(cmd, **kw)


def regenerate() -> None:
    """Rebuild the notebook from its generator, which validates it on the way out."""
    rc = subprocess.run([sys.executable, "sprint/_make_notebook.py"], cwd=REPO).returncode
    if rc != 0:
        sys.exit(f"notebook generation/validation failed (rc {rc}); not pushing")


def cmd_push(a) -> int:
    regenerate()
    overrides = {k: getattr(a, k) for k in KNOBS}
    applied = apply_config(NB, overrides)
    # revalidate: the injected values go through the same checks as generated ones
    if subprocess.run([sys.executable, "sprint/_check_notebook.py"], cwd=REPO).returncode != 0:
        sys.exit("validator rejected the notebook after config injection; not pushing")

    stage = Path(a.stage)
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NB, stage / NB.name)
    meta = {
        "id": f"{username(required=not a.dry_run)}/{a.slug}",
        "title": a.title,
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        # Prize terms: the discovery stays unpublished until the prize is announced, and a
        # public notebook publishes its outputs. This must never flip to false.
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nstaged {stage}")
    print(f"  id          {meta['id']}")
    print(f"  private     {meta['is_private']}   gpu {meta['enable_gpu']}   "
          f"internet {meta['enable_internet']}")
    print(f"  accelerator {a.accelerator}")
    print(f"  config      {applied or '(notebook defaults)'}")
    if a.dry_run:
        print("\n--dry-run: nothing pushed. Drop it to commit and queue the run.")
        return 0

    cmd = kaggle_cmd() + ["kernels", "push", "-p", str(stage)]
    if a.accelerator:
        cmd += ["--accelerator", a.accelerator]
    if a.timeout:
        cmd += ["-t", str(a.timeout)]
    rc = run(cmd).returncode
    if rc == 0:
        print(f"\nqueued. Watch it with:\n"
              f"  python sprint/kaggle_sync.py status\n"
              f"and when it says complete:\n"
              f"  python sprint/kaggle_sync.py output")
    return rc


def cmd_status(a) -> int:
    return run(kaggle_cmd() + ["kernels", "status", f"{username()}/{a.slug}"]).returncode


def cmd_pull(a) -> int:
    """Fetch whatever is on Kaggle now - including cells edited in the browser."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = run(kaggle_cmd() + ["kernels", "pull", f"{username()}/{a.slug}", "-p", str(out), "-m"]).returncode
    if rc != 0:
        return rc
    remote = next(iter(sorted(out.glob("*.ipynb"))), None)
    if remote and NB.exists():
        same = json.loads(remote.read_text(encoding="utf-8")) == json.loads(NB.read_text(encoding="utf-8"))
        print(f"\n{remote}")
        print("identical to sprint/kaggle_sweep.ipynb" if same else
              "DIFFERS from sprint/kaggle_sweep.ipynb - the browser copy was hand-edited. "
              "Reconcile before the next push, which overwrites it.")
    return 0


def cmd_output(a) -> int:
    out = Path(a.out) / a.slug
    out.mkdir(parents=True, exist_ok=True)
    rc = run(kaggle_cmd() + ["kernels", "output", f"{username()}/{a.slug}", "-p", str(out)]).returncode
    if rc == 0:
        files = sorted(p for p in out.rglob("*") if p.is_file())
        print(f"\n{len(files)} file(s) under {out}")
        for p in files[:25]:
            print(f"   {p.relative_to(out)}  {p.stat().st_size / 1e6:.1f} MB")
    return rc


def cmd_quota(a) -> int:
    return run(kaggle_cmd() + ["quota"]).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", default=SLUG)
    ap.add_argument("--user", help="Kaggle account name, if it cannot be resolved automatically")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("auth", help="store a token (or run the OAuth flow) and verify it")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--token", help="the value copied from kaggle.com/settings -> API")
    g.add_argument("--login", action="store_true", help="browser OAuth; no token to copy")
    p.set_defaults(fn=cmd_auth)

    p = sub.add_parser("push", help="regenerate, validate, commit and queue a run")
    p.add_argument("--what", choices=("control", "pherc0800"))
    p.add_argument("--seg")
    p.add_argument("--dmin", type=int)
    p.add_argument("--dmax", type=int)
    p.add_argument("--models")
    p.add_argument("--villa-par", type=int, dest="villa_par")
    p.add_argument("--hec-par", type=int, dest="hec_par")
    p.add_argument("--title", default=TITLE)
    p.add_argument("--accelerator", default="NvidiaTeslaT4",
                   help="NvidiaTeslaT4 | NvidiaTeslaP100 | Tpu1VmV38. T4 is the two-card "
                        "shape sweep_dual.py is written for.")
    p.add_argument("--timeout", type=int, help="seconds; caps the run")
    p.add_argument("--stage", default=str(REPO / "private" / "kaggle_push"))
    p.add_argument("--dry-run", action="store_true",
                   help="stage and validate, push nothing. A push costs quota; this does not.")
    p.set_defaults(fn=cmd_push)

    p = sub.add_parser("status", help="is the run queued, running, complete or failed")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("pull", help="download the notebook as it exists on Kaggle, and diff it")
    p.add_argument("--out", default=str(REPO / "private" / "kaggle_pull"))
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser("output", help="download /kaggle/working from the last run, plus its log")
    p.add_argument("--out", default=str(REPO / "private" / "my_data" / "kaggle"))
    p.set_defaults(fn=cmd_output)

    p = sub.add_parser("quota", help="remaining weekly GPU hours")
    p.set_defaults(fn=cmd_quota)

    a = ap.parse_args(argv)
    if a.user:
        os.environ["KAGGLE_USERNAME"] = a.user
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
