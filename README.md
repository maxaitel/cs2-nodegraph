# OpenCS2 Play Graph

This project builds a lightweight play graph from
[`blanchon/opencs2_dataset`](https://huggingface.co/datasets/blanchon/opencs2_dataset).
It does not download the full video dataset. It queries the dataset's remote
Parquet event/index tables and, by default, reads only a capped number of small
per-POV tick sidecars to infer flick-like events.

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
