# OpenCS2 Play Graph

This project builds a lightweight play graph from
[`blanchon/opencs2_dataset`](https://huggingface.co/datasets/blanchon/opencs2_dataset).

## Data Status

This repository does **not** contain the full OpenCS2 dataset.

What is included:

- A generated sample graph in `webui/public/data/`.
- The committed sample has `1,200` play nodes and `5,751` similarity edges.
- Each node is an individual kill/play event enriched with round context.

What is not included:

- The full OpenCS2 media corpus.
- The full set of POV MP4 videos.
- The full set of per-POV tick sidecars.
- A full-graph export for every kill in the dataset.

The web UI works out of the box for the committed sample data. When you select a
node, it streams that POV video from Hugging Face on demand; the video is not
stored in this repo.

The graph builder queries the dataset's remote Parquet event/index tables and,
by default, reads only a capped number of small per-POV tick sidecars to infer
flick-like events. This keeps local setup small, but it means the checked-in UI
data should be treated as a sample, not as the full corpus.

## Run

```bash
uv run --python 3.12 python scripts/build_opencs2_graph.py
```

Main outputs are written to `outputs/`:

- `opencs2_play_graph.html` - interactive nearest-neighbor graph.
- `opencs2_play_vectors.csv` - sampled plays, labels, features, and 2D coordinates.
- `opencs2_play_edges.csv` - graph edges between similar plays.
- `opencs2_play_summary.json` - run settings and label counts.

Useful options:

```bash
uv run --python 3.12 python scripts/build_opencs2_graph.py \
  --max-events 2000 \
  --max-tick-events 500 \
  --neighbors 8
```

Use `--skip-ticks` when you want a metadata-only run with no tick-sidecar reads.

To build a much larger graph from the remote event tables, raise the caps:

```bash
uv run --python 3.12 python scripts/build_opencs2_graph.py \
  --max-events 111087 \
  --max-tick-events 0 \
  --skip-ticks
```

That scans the full current kill-event table but still does not download videos.
Very large graphs may be too heavy for the browser UI without additional
aggregation or server-side filtering.

## Download Dataset Files

Use the downloader when you actually want local dataset files. The safe default
downloads only metadata/event/index/static files:

```bash
uv run --python 3.12 python scripts/download_opencs2_dataset.py
```

Download event/index metadata plus all tick sidecars, but still no videos:

```bash
uv run --python 3.12 python scripts/download_opencs2_dataset.py --mode sidecars
```

Download the full Hugging Face snapshot, including MP4 videos:

```bash
uv run --python 3.12 python scripts/download_opencs2_dataset.py --mode full --yes
```

Full mode can be very large. Downloaded files go to `data/opencs2_dataset/`,
which is gitignored.

## Web UI

The `webui/` app uses Cytoscape.js rather than a custom graph renderer.
Selecting a graph node loads that play's POV video in the side panel and cues it
to a few seconds before the kill/event timestamp.

```bash
cd webui
npm install
npm run import:data
npm run dev
```

Use `npm run import:data` again after regenerating files in `outputs/`.

## Labels

The dataset directly exposes clutch and 1v1 kill context. Ace labels are derived
from player-round kill totals. Impressive multikills are stricter than simple
multi-kills: they require pace, a four-plus kill round, or a three-plus kill
round with extra quality/context such as headshot, wallbang, smoke, noscope, or
clutch.

Flick is inferred, not ground truth. For sampled tick sidecars, the script
measures view-angle movement shortly before the kill and tags the strongest
snap-aim events as `flick`.
