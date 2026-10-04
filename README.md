# Deep Learning with Validated Explanations for Hyperspectral Classification of SWD-Infested Postharvest Blueberries

Source code for the study of hyperspectral, spatial-spectral deep learning
(3D-CNN and 3D-CNN-Transformer) for non-destructive classification of
Spotted Wing Drosophila (SWD) infestation in blueberries, with an
explanation-validation framework based on faithfulness, attribution stability,
and IG-SHAP cross-method agreement.

> **Paper:** *Computers and Electronics in Agriculture* (under revision).
> Citation details will be added on acceptance; see `CITATION.cff`.

## What is in this repository

- `swd_detection/` — the full analysis package: shard creation, dataset
  loaders, the two model definitions (`cnn3d`, `cnn3d_transformer`), training,
  evaluation, cross-stage generalization, attribution-guided band selection,
  the explanation-validation analyses, and `make_figures.py`, the single script
  that generates all 3-seed manuscript figures.
- `scripts/` — `make_demo_data.py` (builds the tiny demo dataset), plus
  `aggregate_seeds.py` and `mcnemar_compare.py` for the 3-seed workflow.
- `segmentation/` — `segmentation.py`, the classical (Otsu + morphology)
  berry segmentation used by `create_segmented_shards.py` to build the
  mean-spectra inputs behind the mean-spectra figure.
- `data_demo/` — a reduced demo subset for a functional demo run.

## Scope of reproducibility

This repository provides the **code** and a **tiny demo dataset** for verifying
that the pipeline runs end to end. The demo subset is deliberately small and
spatially reduced, so it **does not reproduce the paper's reported numbers**. The
full model-ready hyperspectral dataset is large and is available from the
corresponding author on reasonable request.

## How to run

The steps below are the end-to-end run order. Each links to the section that
gives the exact command, so nothing here is repeated in full.

1. **Install** the dependencies once — see [Installation](#installation).
2. **Build the shards.** Everything downstream reads per-cell shards, not raw
   boards. `python swd_detection/create_shards.py` writes the full shard set and
   its `manifest.csv`; add `create_segmented_shards.py` for the mean-spectra
   inputs.
3. **Run the demo** on the shipped demo data — see
   [Demo run](#demo-run).
4. **Run the full study** once per seed (42, 43, 44) — see
   [Three-seed analysis](#three-seed-analysis-seeds-42-43-44).
5. **Aggregate** the three seed trees into mean ± SD, and optionally run the
   McNemar paired tests — same section as step 4.
6. **Build the figures** from the aggregated runs — see
   [Manuscript figures](#manuscript-figures).

Steps 3 through 6 are all that is needed once the shards from step 2 exist.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Demo run

The demo folder (`data_demo/shards/`) already contains a manifest and 320
spatially reduced hyperspectral cell shards. From the repository root:

```bash
python swd_detection/run_all.py \
    --demo_run \
    --shard_dir data_demo/shards \
    --sensor nir --model cnn3d --mode stage_agnostic \
    --skip_multispectral
```

The `--demo_run` flag runs the demo as a short 3-epoch training pass with a
reduced attribution analysis. `--skip_multispectral` keeps the public demo
independent of the full raw-data folders. The command should complete in a few
minutes on a CPU and confirms that shard loading, training, evaluation, and
explanation all run.

## Rebuilding the demo data from the full dataset

The committed `data_demo/` was produced from the full shard set with:

```bash
python scripts/make_demo_data.py \
    --source-manifest /path/to/full/shards/manifest.csv \
    --out data_demo/shards
```

`manifest.csv` for the full shards is produced by
`python swd_detection/create_shards.py`. See that script's header for the
expected raw-data layout and the environment variables that point to it
(`SWD_DATA_ROOT`, `SWD_SHARD_DIR`, and the per-class path variables in
`swd_detection/config.py`).

## Reproducing the full study

With the full shard set in place and `--shard_dir` pointing at it, the complete
set of experiments (both sensors, both models, stage-agnostic and stage-specific
training, cross-stage generalization, attribution-guided band selection, and the
explanation-validation analyses) is driven by `swd_detection/run_all.py`. Run
`python swd_detection/run_all.py --help` for the full option list.

The two architectures evaluated are the 3D-CNN (`cnn3d`) and the hybrid
3D-CNN-Transformer (`cnn3d_transformer`), on the NIR and VNIR sensors.

## Three-seed analysis (seeds 42, 43, 44)

All results are reported as mean ± SD across three training seeds. The
board-level train/validation/test split is held fixed at the partition seed
(`config.SEED = 42`) so the split never changes; only weight initialization,
dropout, augmentation, and shuffling vary between seeds. The training seed is
set with the `--seed` argument:

```bash
python swd_detection/run_all.py --shard_dir <shards> --seed 42 --output_dir outputs_seed42
python swd_detection/run_all.py --shard_dir <shards> --seed 43 --output_dir outputs_seed43
python swd_detection/run_all.py --shard_dir <shards> --seed 44 --output_dir outputs_seed44
```

Each seed writes its own output tree. Aggregate the three into mean ± SD with:

```bash
python scripts/aggregate_seeds.py outputs_seed42 outputs_seed43 outputs_seed44 --out seed_summary
```

Paired significance tests on the fixed test set (no extra training) use
McNemar's test on any two runs' per-sample prediction CSVs:

```bash
python scripts/mcnemar_compare.py <run_A>_results.csv <run_B>_results.csv
```

## Plots

The paper's 3-seed figures (full-spectrum and selected-wavelength confusion
panels, stage-specific performance, cross-stage accuracy, selected wavelengths,
attribution stability, IG-SHAP cross-method agreement, and mean spectra) are all
produced by a single script, `swd_detection/make_figures.py`, from the three
per-seed output trees:

```bash
python swd_detection/make_figures.py \
    --root outputs_paired_3seed \
    --seed42-full-roots outputs_paired_cnn3d outputs_paired_cnn3d_transformer \
    --out-dir outputs/manuscript_figures_paired_3seed
```

Run `python swd_detection/make_figures.py --help` for the full list of input
roots (cross-stage, mean-spectra, and cross-method CSV paths). It reports mean ±
SD across the available seeds for each plotted condition.

## License

MIT. See `LICENSE`.
