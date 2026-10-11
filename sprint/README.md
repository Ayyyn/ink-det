# 7-day First Letters sprint (from 2026-10-09)

Replicating the ink stage of the winning PHerc0343 submission on scrolls that are still
eligible. Everything here is open source; findings are not (see the prize rule below).

## Why these scripts exist

On 2026-10-09 Erwin Nieuwlaar won $50K for PHerc0343 and published his whole method,
pinned and bit-reproducible, at `github.com/Nieuwlaar/pherc343-first-letters`. The
chain is: spiral fit -> snap to the m7 surface -> lasagna flatten -> join two windings
-> render 66 layers -> **four ink models averaged as z-scores over 31 depths** -> image.

Two facts make a fast replication possible:

1. **PHerc0800 already has published renders.** Its six auto-grown segments are rendered
   on its eligible volume `20250521135224` and ship 31-layer surface volumes. Steps 1-5
   of that chain are done and public; only inference is left. No VC3D, no spiral fit.
2. **Only the ink ensemble was ever the missing piece.** Single checkpoints have
   certainly been run over these public renders. One stretch per model, fixed across
   depths, then a four-model z-score average, is what turned a null into letters.

## What these scripts change, and nothing else

| script | purpose |
|---|---|
| `setup_gpu.sh` | villa at the pinned commit `0fb38c45`, CPython 3.14, the four public checkpoints (sha256-verified), his repo, the N_LAYERS patch |
| `pull_render.py` | copy one published surface volume from anonymous S3 to local disk |
| `sweep31.sh` | his `run_depth_sweep.sh` with `N_LAYERS=31` and no padded second run |
| `ensemble31.sh` | his `ensemble_maps.py` over a 31-layer sweep |
| `patch_n_layers.py` | makes `ensemble_maps.py` read `N_LAYERS` instead of hardcoding 66 |
| `kaggle_ink_0800.ipynb` | the whole thing as a free-tier Kaggle notebook |

**The 31-layer arithmetic.** The villa models take a 17-layer window `s..s+16` with
`s = N/2 + d - 8`, so 66 layers admit `d = -25..+24` and 31 layers admit **`d = -7..+7`**.
Hecate centres on `(N-1)/2 + d = 15 + d`. The patch is backward compatible: at `N=66`
both expressions reduce to his originals (`d+25`, `32.50`), so his reproduction is
untouched.

**Why the padded run is dropped.** His segment is two windings joined at the spiral seam,
and the models work on tiles, so he ran everything twice — plain and zero-padded — and
took each half from the run where it sat at its own mesh's tile phase. A single
auto-grown segment has no seam. Dropping it halves the GPU time.

**Why both directions, always.** On an unread scroll the sheet's orientation is unknown,
so `--direction both` is not optional. It also gives the falsification test for free: a
real text row shows on one face only, so anything equally strong on the `_blind` side is
an artefact.

## Run it

Free, starts now (16 GB is enough — the models tile):

> Kaggle: `kaggle_sweep.ipynb`, **GPU T4 x2** and **Internet On** (needs a
> phone-verified account), **private**, run top to bottom. Or drive it from a laptop
> with `kaggle_sync.py` and never open the browser - see below.

On a rented 24 GB box (Vast.ai / RunPod, Linux, ~$0.30-0.60/hr):

```bash
git clone https://github.com/Ayyyn/ink-det.git && cd ink-det
bash sprint/setup_gpu.sh /data                       # ~15 min
/data/venv/bin/python sprint/pull_render.py --scroll PHerc0800 \
    --segment 20251028222030 --out /data/renders     # ~1 min, 200 MB
P_VILLA=2 P_HEC=2 bash sprint/sweep31.sh /data/renders/20251028222030.zarr \
    /data/maps/20251028222030 -3 3                   # widen to -7 7 after
bash sprint/ensemble31.sh /data/renders/20251028222030.zarr \
    /data/maps/20251028222030 /data/images/20251028222030 '-3..3' '-1 0 1'
```

The sweep is resumable: existing maps are skipped, and depths run from `d = 0` outwards,
so an interrupted run still holds the depths that matter most.

### Driving Kaggle from here instead of retyping cells

Every failure so far on Kaggle has been transport, not science: a shell command pasted
into a Python cell, a backslash continuation the `!` magic silently ignores, a hand-edited
cell drifting from the repo. `kaggle_sync.py` removes the retyping step. Put a token at
`~/.kaggle/kaggle.json` (kaggle.com/settings -> API -> Create New Token), then:

```bash
python sprint/kaggle_sync.py push --what pherc0800 --seg 20251028222030 --dmin -7 --dmax 7
python sprint/kaggle_sync.py status
python sprint/kaggle_sync.py output          # /kaggle/working + the run log
```

It regenerates the notebook from `_make_notebook.py`, rewrites the config cell from the
flags, re-runs `_check_notebook.py`, and pushes with `--accelerator NvidiaTeslaT4`.

Three properties to keep in mind:

- **A push is a run.** The API has no save-without-execute, so each push bills against the
  weekly GPU quota. Use `--dry-run` to check the plan for free.
- **The run is headless**, which suits a 40-minute sweep: it survives a closed laptop.
  Nothing is visible until it ends, so evidence must be written to `/kaggle/working` -
  which cell 10 of the notebook does.
- **Fixes to `sprint/*.py` do not travel through the notebook.** It clones this repo from
  GitHub, so a script fix reaches Kaggle by pushing to `origin`. Only notebook changes
  need `kaggle_sync.py push`.

`pull` fetches the copy that is live on Kaggle and says whether it differs from the repo,
which is the check to run before a push overwrites browser edits.

## Target order

| target | state | what it costs |
|---|---|---|
| **PHerc0800** | 6 published segments on the eligible volume, 31-layer renders | inference only, hours |
| **PHerc0268**, **PHerc1218** | 8.64 um, m7 surface **and** lasagna published, no segments | the full spiral chain, 24 GB box, VC3D |
| 0175A/B, 0306B, 0483A/B, 0490A/B | 8.64 um, **surface only, no lasagna** | lasagna first, then the chain |
| the 9.362 um eleven | lower resolution tier | last |

**PHerc0800 is fewer shots than it looks.** Pairwise median nearest-neighbour distance
between the six meshes (level-0 voxels, 1 vox = 8.64 um):

```
                 213516  220042  220955  222030  225813  010146
20251028213516        0      23      43      20      31      47
20251028220042       23       0      25      41      51      65
20251028220955       43      25       0      62      75      89
20251028222030       19      38      58       0      16      33
20251028225813       30      50      73      17       0      22
20251029010146      207     195     210     296     218       0
```

Wrap pitch on these scrolls is 300-600 um (35-70 voxels) and a sheet is ~150 um
(17 voxels). So `{213516, 220042, 222030, 225813}` are all within about one sheet
thickness of each other — **one sheet, grown four times** — `220955` is roughly a wrap
away, and `010146` is a few wraps away but only 0.72 cm2. Treat PHerc0800 as **one good
patch (`222030`, the largest at 4.97 cm2) plus a second wrap**, not six chances. Start
0268/1218 in parallel on day one rather than waiting for 0800 to fail.

## Prize rules that bind this work

- **Do not publish findings.** `34_prizes.md:229`: the discovery must not be made public
  until the prize is announced. Code here is public; ink maps, candidate letters and
  submission drafts are gitignored (`private/`, `out/`, `ink/`, `renders/`).
- **Code may stay private until you accept a prize** (`34_prizes.md:115`) — so a public
  repo is a choice, not an obligation. We publish the tooling because it is separately
  Progress-Prize eligible.
- **No training on the target.** None of the four models was trained on PHerc0800.
  Keep it that way, or the submission needs k-fold evidence instead.
- **Non-hallucination is now the burden of proof** (the 0.5 mm window cap was dropped on
  2026-09-10). The `_blind` side and a held-out region are the cheap evidence.
