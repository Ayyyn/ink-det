# Vesuvius Challenge — First Letters campaign

Project context for Claude Code. Drop this at the repo root; it loads every session.

## Mission

Win a **First Letters prize**: 10 visible, legible letters within a single **4 cm²**
area on one of the 13 eligible Herculaneum scrolls. **$50,000 per scroll, 10 prizes
available, deadline 25 June 2027.** Secondary and continuous: monthly **Progress
Prizes** ($500–$20,000, guaranteed $20,000 for best-of-month) for open-source
contributions against the challenge's published open-problems list.

Operator: Ayushi Nanavati. Worked with the Vesuvius Challenge team as an image
segmentor for 2 months in 2023; competed for First Letters in 2024 on ink detection
and did not win. This is a side project at roughly 20 hrs/week against other
commitments — scope decisions should assume interruption, not continuity.

## The strategic thesis

Three claims drive every decision here. Argue with them if evidence contradicts them,
but do not quietly drift from them.

1. **Ink detection is no longer the bottleneck; geometry is.** There are published
   pretrained ink checkpoints and a documented end-to-end recipe. The challenge's own
   2026 open-problems writeup ranks the top bottlenecks as label quality/surface
   supervision, compressed-region X-ray scatter, mesh topology repair, and fiber
   tracing — all upstream of ink. The 2024 loss was on the wrong axis.

2. **4 cm² is the exploit.** The Grand Prize needs 100% of a scroll's recto at 70%
   per-column legibility. First Letters needs a 2 cm × 2 cm patch — order of a
   thousandth of the same scroll. **We do not need a working global pipeline. We need
   one clean patch, once.** Anything framed as "improving the pipeline generally" is
   out of scope unless it also ships as a Progress Prize submission.

3. **The job is search, not construction.** Across 13 scrolls, find the single easiest
   4 cm² region and hammer it. Effort goes to either (a) cheaply ranking candidate
   regions, or (b) attacking the top-ranked one.

## Scroll shortlist

All 13 eligible scrolls are scanned, ship a published recto surface prediction and a
lasagna prediction, and **none is segmented, inked, or read.** Ranked (data-browser
index dated 2026-06-15):

| target | scroll | why |
|---|---|---|
| 1 | **PHerc1203** | Only eligible scroll with a **2.403 µm** scan, plus a surface prediction on that volume. Browser score 7.6 vs 0.6–1.4 for all others. |
| 2 | **PHerc1447** | 15 existing segments, warmest geometry, 8.64 µm. Also the worked example in the official VC3D tutorial. |
| 3 | **PHerc0800** | 6 existing segments, same 8.64 µm / 116 keV scan batch as 1447. |
| 4 | **PHerc0826** | The one scroll with a documented spiral-fitting dataset in the repo. |

Remaining nine (0125, 0191, 0211, 0257, 0268, 0358, 0813, 1218, 1545): 9.362 or
8.64 µm, no existing segments, nothing distinguishing.

**Why resolution decides the ranking.** The current ink model `ink_canonical_2um`
works on 2.4 µm data; everything runnable on the other twelve is the weaker 9 µm
cross-scroll model. More importantly, **direct 3D ink segmentation** — finding ink in
the volume without first producing a perfect flattened surface — is demonstrated at
2.4 µm on PHerc. Paris 4 (`ink_3d_dino_guided`). That route partly steps around the
geometry bottleneck.

**Open question, blocks target order:** at 0.2 m propagation, PHerc1203's 2.4 µm scan
may cover only a slab rather than the whole roll. `check_1203.py` answers this from
Zarr metadata. If it is a thin slab, the 2.4 µm play is confined to that band and
PHerc1447 becomes primary.

## Where the truth lives

Verify against **source in the `villa` clone**, not blog posts or summaries:

- `scrollprize.org/docs/34_prizes.md` — authoritative prize terms and submission rules
- `scrollprize.org/docs/37_2026_open_problems.md` — the open-problems list, far more
  detailed than the rendered page; this is the wish list Progress Prizes score against
- `scrollprize.org/docs/06_tutorial_VC3D.md` — geometry walkthrough (uses PHerc1447)
- `scrollprize.org/docs/07_tutorial5.md` — ink detection walkthrough (uses PHerc0139 w035)
- `scrollprize.org/docs/38_tutorial_spiral.md` — spiral fitting
- `vesuvius/docs/` — ink_detection.md, accessing_data.md, inference.md, training_flow.md
- `scrollprize.org/src/data/prizeEligibility.json` — the eligible-volume list
- `scrollprize.org/static/data_browser/index.json` — per-scroll assets, S3 URIs, status

Prize terms have changed mid-cycle before. Re-check `34_prizes.md` before any submission.

## Toolchain map (`ScrollPrize/villa`)

| component | role |
|---|---|
| `volume-cartographer` (VC3D) | semi-automatic surface tracing; the team's current approach. Prebuilt: `ghcr.io/scrollprize/villa/volume-cartographer:edge` |
| `vesuvius` | Python library: data access, training, inference, rendering. `vc_render_tifxyz` lives here |
| `lasagna` | PyTorch mesh growing/refinement + flattening; alternative to VC3D GrowPatch |
| `spiral-fitting` | fully automatic global unwrapping by fitting an Archimedean spiral |
| `dinovol` | self-supervised 3D DINO on CT volumes |
| `ink-detection/optimized_inference` | containerized 2.4 µm ink inference |
| `segmentation/` | evaluation, proofreading, model optimization |

Checkpoints on Hugging Face under `scrollprize/`: `ink_9um` (cross-scroll 9 µm, two
seeds × seven steps), `ink_canonical_2um`, `ink_3d_dino_guided`, `surface_m7_nnunet`,
`surface_recto_3dunet`, `copy_displacement_latest`, `fiber_hz_vt`,
`dinovol_v2_ps8_with_paris4_352500`, and `PHerc.1667-iteration-0`…`-5` (the released
six-stage iterative pseudo-labeling trajectory — study this; it is the loop that wins).

## Hard-won gotchas

- **Python 3.14 exactly.** `vesuvius` pins `>=3.14,<3.15`. Let `uv` fetch the
  interpreter; do not fight the system Python.
- **Never download a volume.** Volumes are multi-TB. `vc_render_tifxyz --remote-url`
  caches only the chunks a segment touches. Disk is for renders and caches.
- **S3 is anonymous.** `s3://vesuvius-challenge-open-data/` — every call needs
  `--no-sign-request`. Co-locate compute with the bucket region.
- **Ink predictions look washed out.** Label smoothing 0.5 puts training targets at
  0.25 background / 0.75 ink. Rescale `(p - 0.25) / 0.5`, clipped to [0,1], before
  concluding a model found nothing.
- **Models are depth-sensitive.** If a checkpoint responds weakly, shift the window
  with `--layer-start` / `--layer-end`, or average nearby windows.
- **On an unread scroll the surface orientation is unknown** — always `--direction both`.
- **Try several checkpoints.** Two seeds × seven steps behave differently per segment.
  This is cheap and it matters.
- **`spiral-fitting`: `lasagna_scale` must match the actual downsample factor** of the
  chosen OME-Zarr group for *that specific scroll*. Read it from the store's `.zattrs`.
  A mismatch silently reads wrong-resolution normals or throws an empty-z-ROI error.
- Long renders and inference runs go in `tmux`.

## Current phase

**Phase 0 — reproduce.** Two exit tests, both required before touching an unread scroll:

- **Track A (geometry):** one clean `.tifxyz` patch grown on PHerc1447 via VC3D.
- **Track B (ink):** legible Greek recovered from PHerc0139 segment w035, which is in
  the ink models' own training set. Its rendered surface volume is already published —
  skip the render the first time.

Then: Phase 1 region scoreboard (weeks 2–5), Phase 2 attack top patches (weeks 6–20),
Phase 3 submit and iterate (weeks 21–42).

## Working agreements

- **Submit early and often.** No penalty for a rejected First Letters submission and no
  reason to wait for June. Feedback beats polish.
- **Kill discipline: three weeks per patch,** then move down the ranked list.
  Abandoning a patch is a decision, not a defeat.
- **Everything ships open source** under a permissive license — it is a prize condition,
  and the recon tooling is separately prize-eligible on its own.
- **Prepare submission requirements in parallel, not at the end:** Docker
  reproducibility, held-out validation with no train/predict overlap, documented
  false-positive mitigation, scale bar on every image, fiber-alignment verification.
- **Don't rebuild what exists.** Check recent Progress Prize writeups and the villa repo
  before building tooling — ScrollFiesta, Will Stevens' unwrapping pipeline and others
  ship regularly.
- **Push back on the plan when evidence contradicts it.** The three thesis claims above
  are load-bearing; if data says one is wrong, say so directly rather than working
  around it.

## Style

Direct and concise. Lead with the finding, not the process. Flag uncertainty explicitly
rather than hedging everything. When something in this file goes stale, say so.
