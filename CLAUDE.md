# Vesuvius Challenge — First Letters campaign

Project context for Claude Code. Lives at the repo root; it loads every session.

Last verified against source: **2026-09-18** (villa clone at `./villa`, pulled that day). Facts marked
**[V]** were checked against repo source or S3 metadata on that date, not from memory.

## Mission

Win a **First Letters prize**: 10 visible, legible letters within a single **4 cm²**
area on one of the **23** eligible Herculaneum scroll volumes. **$50,000 per scroll, 10
prizes available, deadline 25 June 2027.** **[V]** `34_prizes.md`

Secondary and continuous: monthly **Progress Prizes** ($500–$20,000, guaranteed
$20,000 for best-of-month) for open-source contributions against the challenge's
published open-problems list. **Next deadline: 11:59pm Pacific, 30 September 2026.**
**[V]** `34_prizes.md`

Operator: Ayushi Nanavati. Worked with the Vesuvius Challenge team as an image
segmentor for 2 months in 2023; competed for First Letters in 2024 on ink detection
and did not win. This is a side project at roughly 20 hrs/week against other
commitments — scope decisions should assume interruption, not continuity.

## The strategic thesis

Three claims drive every decision here. Argue with them if evidence contradicts them,
but do not quietly drift from them.

1. **Ink detection is no longer the bottleneck; geometry is.** The challenge's 2026
   open-problems writeup ranks the top bottlenecks as label quality/surface
   supervision, compressed-region X-ray scatter, mesh topology repair, and fiber
   tracing — all upstream of ink. The 2024 loss was on the wrong axis. This claim got
   *stronger* on 2026-09-07: the direct-3D-ink escape hatch turned out to be
   unavailable on eligible data (see "The eligibility constraint"), so there is no way
   around geometry.

2. **4 cm² is the exploit.** The Grand Prize needs 100% of a scroll's recto at 70%
   per-column legibility. First Letters needs a 2 cm × 2 cm patch. **We do not need a
   working global pipeline. We need one clean patch, once.** Anything framed as
   "improving the pipeline generally" is out of scope unless it also ships as a
   Progress Prize submission. This claim has already paid once — it defeated the old
   "thin slab disqualifies a scroll" rule, which was wrong because a slab many times
   larger than 4 cm² is not a constraint at all.

3. **The job is search, not construction.** Across 23 scrolls, find the single easiest
   4 cm² region and hammer it. Effort goes to either (a) cheaply ranking candidate
   regions, or (b) attacking the top-ranked one.

## The eligibility constraint — read this before planning anything

**[V]** `prizeEligibility.json` pins **one specific volume id per scroll**. First Letters
scores against its own list, **`first-letters-2027` (23 volumes)**; the Grand Prize keeps
`grand-prize-2027` (13 volumes). The First Letters set is a superset: the original 13 plus
10 added 2026-09-10. **[V]** `34_prizes.md:204`

That this is volume-level and deliberate is confirmed by contrast: the PHerc. Paris 4
Title Prize explicitly says "using any of its scans — *including the 2.4 µm volumes*."
Where a high-res carve-out exists, they spell it out. First Letters has none.

**Consequence: every eligible volume is 8.64 or 9.362 µm.** **[V]**

- `ink_canonical_2um` and `ink_3d_dino_guided` operate at 2.4 µm and therefore **do not
  apply to any eligible target.** The direct-3D-ink route is off the table for this prize.
- The only applicable ink checkpoint family is **`ink_9um`** (cross-scroll, two seeds ×
  seven steps).
- PHerc1203's 2.403 µm volume (`20260319130212`) is **not eligible**. Its eligible
  volume is `20250820131727` at 9.362 µm. (For the record, that 2.4 µm scan is also a
  36.4 mm slab of a 177.7 mm roll — but the slab was never the disqualifier.)

**Volume-locked: settled.** On 2026-09-10 First Letters got its *own* pinned list,
`first-letters-2027` (23 volumes), separate from `grand-prize-2027` (still 13). Creating
a second volume-pinned list confirms eligibility is volume-level, not scroll-level. The
strict reading was correct; no Discord answer needed. **[V]** `prizeEligibility.json`

## Scroll shortlist

All 23 First Letters volumes ship a published recto surface prediction (lasagna for
14 of 23); **none has been read.** **[V]** `scripts/eligible_assets.py --prize
first-letters-2027`

Twelve are at the top resolution tier (8.64 µm); eleven at 9.362 µm.

| tier | scrolls | px | segs | why |
|---|---|---|---|---|
| 1 | **PHerc1447** | 8.64 | 15 | Most existing geometry; the worked example in the official VC3D tutorial. |
| 2 | **PHerc0800** | 8.64 | 6 | Second-most geometry, same 8.64 µm scan batch as 1447. |
| 3 | **PHerc0175A, 0175B, 0306B, 0343, 0483A, 0483B, 0490A, 0490B** | 8.64 | 0 | **Newly eligible 2026-09-10, top resolution tier, untouched by anyone.** Best hunting ground: no competitor has a head start. |
| 4 | PHerc0268, PHerc1218 | 8.64 | 0 | Top tier, no geometry. |
| 5 | PHerc0846A, 0846B (new), 0125, 0191, 0211, 0257, 0358, 0813, 0826, 1203, 1545 | 9.362 | 0 | Lower resolution tier. |

**Do not rank by the data browser's `score` field.** It is attached to the *scroll*,
not the eligible volume. PHerc1203 scores 7.6 there purely because of its 2.4 µm scan,
which we cannot use; on eligible data it is a bottom-tier target. This single confusion
inverted the entire original target order.

## Where the truth lives

Verify against **source in the `villa` clone**, not blog posts or summaries:

- `scrollprize.org/docs/34_prizes.md` — authoritative prize terms and submission rules
- `scrollprize.org/src/data/prizeEligibility.json` — the pinned eligible-volume list
- `scrollprize.org/docs/37_2026_open_problems.md` — the open-problems list, far more
  detailed than the rendered page; this is the wish list Progress Prizes score against
- `scrollprize.org/docs/06_tutorial_VC3D.md` — geometry walkthrough (uses PHerc1447)
- `scrollprize.org/docs/07_tutorial5.md` — ink detection walkthrough (uses PHerc0139 w035)
- `scrollprize.org/docs/38_tutorial_spiral.md` — spiral fitting
- `vesuvius/docs/` — ink_detection.md, accessing_data.md, inference.md, training_flow.md
- `scrollprize.org/static/data_browser/index.json` — per-scroll assets, S3 URIs, status

Prize terms have changed mid-cycle before. Re-check `34_prizes.md` before any submission.

## Submission requirements that constrain the build

**[V]** `34_prizes.md`, First Letters criteria. Design for these from day one:

- **Meshes** in tifxyz, including a low-distortion isometric 2D flattening.
- **Image** generated programmatically from CT volume + submitted mesh. No manual
  character annotation, including annotations used as training data.
- **Window size: no hard cap** (changed 2026-09-10; the old ≤0.5 mm rule is gone).
  "Large windows are fine, but you must demonstrate that the text is not hallucinated —
  for example by finding different, plausible letters on a held-out region." The burden
  is now *evidence of non-hallucination*, not a size limit. **[V]** `34_prizes.md:223`
- **No overlap between training and prediction regions.**
- **Scale bar** showing 1 cm, plus pixel and mm dimensions of representative letters.
- **Name the image after the tifxyz mesh** it was rendered from.
- **Annotate text rows** without obscuring letters; overlay on a fiber-visible
  rendering where possible.
- **Held-out validation** with known ground truth, k-fold if trained on public data.
- **False-positive mitigation** documented explicitly.
- **Do not make the discovery public** until the prize is announced.

**Cheapest possible win, per the docs:** ink is *sometimes visible directly in a
flattened render with no model at all*, usually as bright areas — and that alone
qualifies. Always look at the raw render before reaching for a checkpoint.

## Toolchain map (`ScrollPrize/villa`)

| component | role |
|---|---|
| `volume-cartographer` (VC3D) | semi-automatic surface tracing; the team's current approach. Prebuilt: `ghcr.io/scrollprize/villa/volume-cartographer:edge` |
| `vesuvius` | Python library: data access, training, inference, rendering. `vc_render_tifxyz` lives here |
| `lasagna` | PyTorch mesh growing/refinement + flattening; alternative to VC3D GrowPatch |
| `spiral-fitting` | fully automatic global unwrapping by fitting an Archimedean spiral |
| `dinovol` | self-supervised 3D DINO on CT volumes |
| `ink-detection/optimized_inference` | containerized 2.4 µm ink inference — **not applicable to eligible volumes** |
| `segmentation/` | evaluation (supervised Dice-family metrics vs. ground truth), proofreading, model optimization |

Checkpoints on Hugging Face under `scrollprize/`. Applicable here: **`ink_9um`**,
`surface_m7_nnunet`, `surface_recto_3dunet`, `copy_displacement_latest`, `fiber_hz_vt`,
`dinovol_v2_ps8_with_paris4_352500`, and `PHerc.1667-iteration-0`…`-5` (the released
six-stage iterative pseudo-labeling trajectory — study this; it is the loop that wins).
Not applicable: `ink_canonical_2um`, `ink_3d_dino_guided` (2.4 µm only).

## Hardware reality

**[V]** 2026-09-07. Local box: AMD Radeon integrated graphics, **no NVIDIA GPU**,
13.9 GB RAM, 16 threads, ~534 GB free, Windows 11 + WSL2 Ubuntu + Docker Desktop.

- **Nothing with a CUDA checkpoint runs locally.** Ink inference, nnU-Net surface
  prediction, and dinovol all require rented GPU.
- **But the precomputed surface and lasagna predictions are already published for all
  13 eligible volumes.** Streaming and scoring those is pure CPU work and runs here.
  Phase 1 needs no GPU at all. Defer rental until a patch is chosen.

## Hard-won gotchas

- **Python 3.14 exactly** for `vesuvius` — it pins `>=3.14,<3.15`. Let `uv` fetch the
  interpreter; do not fight the system Python. (Local default is 3.12; the throwaway
  `.venv-check` used for metadata scripts is 3.12 and that is fine — it never imports
  vesuvius.)
- **Never download a volume.** Volumes are multi-TB. `vc_render_tifxyz --remote-url`
  caches only the chunks a segment touches. Disk is for renders and caches.
- **S3 is anonymous.** `s3://vesuvius-challenge-open-data/` — every call needs
  `--no-sign-request`. Co-locate compute with the bucket region.
- **`index.json` is UTF-8 and Windows will not guess it.** Always pass
  `encoding="utf-8"`; the cp1252 default throws on byte 0x81.
- **The bundled `index.json` is stale** (dated 2026-06-15). Re-pull from
  scrollprize.org before trusting asset lists.
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

**Phase 1 — region scoreboard.** Target: **30 September 2026 Progress Prize**, ~23 days.

Stream the published surface/lasagna predictions for all 13 eligible volumes and score
every candidate 4 cm² window for geometry tractability, producing a cross-scroll ranking.
Chosen because it is simultaneously the campaign's own Phase 1, a Progress Prize
submission, and runnable with zero GPU spend.

Progress-Prize thesis, against the open-problems bottleneck table: it supplies
**"scan-quality metrics"** (compressed-region row) and a **"reproducible streaming
pipeline"** (data-scale row), and it feeds villa issue #191 (surface/fiber predictions
in compressed or highly curved areas) by locating those regions automatically.
**[V]** nothing in villa currently does this; `segmentation/evaluation` is supervised
model scoring against ground truth, which is a different problem.

**Phase 0 reproduction tests** — still owed, now GPU-gated, so they follow the rental
decision rather than blocking Phase 1:
- **Track A (geometry):** one clean `.tifxyz` patch grown on PHerc1447 via VC3D.
- **Track B (ink):** legible Greek recovered from PHerc0139 segment w035, which is in
  the ink models' own training set. Its rendered surface volume is already published —
  skip the render the first time.

Then: Phase 2 attack top patches, Phase 3 submit and iterate.

## Working agreements

- **Submit early and often.** No penalty for a rejected First Letters submission and no
  reason to wait for June. Feedback beats polish. Progress Prizes explicitly favor work
  released early in the month.
- **Kill discipline: three weeks per patch,** then move down the ranked list.
  Abandoning a patch is a decision, not a defeat.
- **Everything ships open source** under a permissive license — it is a prize condition,
  and the recon tooling is separately prize-eligible on its own.
- **Prepare submission requirements in parallel, not at the end** — see the constraints
  section above.
- **Don't rebuild what exists.** Check recent Progress Prize writeups, the villa repo,
  and the "help wanted" wishlist before building tooling.
- **Verify against pinned source, and date the claim.** The first version of this brief
  inverted its own target ranking by trusting a summary statistic. Mark checked facts
  **[V]** with the date.
- **Push back on the plan when evidence contradicts it.** The three thesis claims above
  are load-bearing; if data says one is wrong, say so directly rather than working
  around it.

## Style

Direct and concise. Lead with the finding, not the process. Flag uncertainty explicitly
rather than hedging everything. When something in this file goes stale, say so.
