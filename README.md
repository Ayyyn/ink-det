# ink_finder

Recon tooling for the [Vesuvius Challenge](https://scrollprize.org) **First Letters
Prize**: finding a tractable 4 cm² patch on a Herculaneum scroll that has never been
read, using only the challenge's own published data.

The premise is that the hard part is no longer ink detection but **geometry** — and
that First Letters needs one clean 2 cm × 2 cm patch, not a working global pipeline.
So the job is search: rank candidate regions across every eligible scroll cheaply,
then attack the best one.

## What's here

| script | what it does |
|---|---|
| `scripts/eligible_assets.py` | Cross-references the pinned eligible-volume list against the data-browser index, reporting which precomputed assets exist per prize target. Eligibility pins **one volume id per scroll**, so assets published against a different volume of the same scroll are not usable. |
| `scripts/check_1203.py` | Reads Zarr metadata over anonymous S3 to answer whether a scan covers a whole roll or only a slab. |
| `scripts/triage_scrolls.py` | Stage A. Locates each scroll's axis, extent and axis wobble from its published recto surface prediction, at the coarsest pyramid level. Seconds per scroll. |
| `scripts/score_windows.py` | Stage B. Casts radial rays through the surface prediction and scores candidate 4 cm² windows on how well adjacent wraps stay separated. Emits L0 seed coordinates for VC3D GrowPatch. |

Everything runs on a laptop: **no GPU, no volume downloads.** The published recto
surface predictions are streamed chunk-wise from public S3, and the coarse pyramid
levels are a few MB each.

## Why separability

The challenge's [2026 open problems](https://scrollprize.org/2026_open_problems)
writeup names the compressed-region problem — "some regions lose effective
separability between layers" — among the top bottlenecks, and asks for scan-quality
metrics. A patch is growable when a radial ray out from the scroll axis crosses
sheet, gap, sheet, gap at a regular pitch; it is hopeless when wraps merge into
blobs or the prediction goes hollow. Both are measurable with no labels and no
ground truth.

**Pyramid level matters.** At ~276 µm/voxel a papyrus sheet is sub-voxel, so any wrap
count is a binning artefact rather than a measurement. Stage B runs at ~69 µm/voxel,
where a 300–600 µm wrap pitch spans 4–9 voxels. Measured pitch across three scrolls
came out at 357–449 µm, which is physically consistent and is the main evidence the
metric is reading real structure.

## Status and honest limitations

- Wrap-pitch measurement: **validated** against three independent scrolls.
- Window *ranking*: **weak so far.** Scores span only 0.330–0.362 across three
  different scrolls and the regularity term sits at 0.45–0.48 nearly everywhere,
  which points at a systematic floor (likely speckle in the thresholded prediction
  creating spurious ray crossings) rather than genuine per-region variation. Two of
  the four composite terms saturate at 1.0 and currently carry no information.
  Treat the ranking as provisional.

## Usage

```bash
python -m venv .venv && ./.venv/Scripts/python.exe -m pip install s3fs zarr numpy scipy

# what is eligible, and what is already published for it
python scripts/eligible_assets.py --prize first-letters-2027

# stage A: axis, extent, wobble
python scripts/triage_scrolls.py --px-max 8.64 --only-new

# stage B: score 4 cm2 windows, emit GrowPatch seeds
python scripts/score_windows.py --scroll PHerc0343
```

`scripts/eligible_assets.py` needs a clone of
[ScrollPrize/villa](https://github.com/ScrollPrize/villa) for the pinned eligibility
list and data-browser index; pass `--villa /path/to/villa`.

## Licence

MIT. Open-sourcing is a condition of the prizes this supports, and the recon tooling
is separately prize-eligible on its own.
