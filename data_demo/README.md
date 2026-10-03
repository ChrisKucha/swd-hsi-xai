# Demo data

This folder holds a **reduced** subset of hyperspectral cells used to verify
that the pipeline runs. It is **not** the model-ready dataset and does not
reproduce the paper's results.

- Each cell is spatially downsampled and stored as float16 to keep the folder
  small enough for GitHub.
- The committed demo contains 320 cell shards balanced across sensor, split,
  and class.
- `shards/manifest.csv` indexes the demo cells for `ShardedBlueberryDataset`.
- Regenerate it with `python scripts/make_demo_data.py` (see the repo README).

The full model-ready dataset is available from the corresponding author on
request.
