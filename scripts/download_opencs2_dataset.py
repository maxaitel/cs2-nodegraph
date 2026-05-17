#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DATASET_ID = "blanchon/opencs2_dataset"


MODE_PATTERNS = {
    "metadata": [
        "README.md",
        ".gitattributes",
        "events/*.parquet",
        "index/*.parquet",
        "metadata/*.parquet",
        "static/**",
    ],
    "sidecars": [
        "README.md",
        ".gitattributes",
        "events/*.parquet",
        "index/*.parquet",
        "metadata/*.parquet",
        "static/**",
        "rounds/**/ticks.parquet",
    ],
    "full": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OpenCS2 dataset files from Hugging Face with safe defaults."
    )
    parser.add_argument("--repo-id", default=DATASET_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", type=Path, default=Path("data/opencs2_dataset"))
    parser.add_argument(
        "--mode",
        choices=["metadata", "sidecars", "full"],
        default="metadata",
        help=(
            "metadata: event/index tables only; sidecars: also per-POV ticks; "
            "full: everything, including videos"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required for --mode full because it downloads the video corpus.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without contacting Hugging Face.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allow_patterns = MODE_PATTERNS[args.mode]

    print(f"Dataset: {args.repo_id}@{args.revision}")
    print(f"Destination: {args.local_dir}")
    print(f"Mode: {args.mode}")
    if allow_patterns is None:
        print("Patterns: all files, including MP4 videos")
    else:
        print("Patterns:")
        for pattern in allow_patterns:
            print(f"  - {pattern}")

    if args.dry_run:
        print("Dry run only; no files downloaded.")
        return

    if args.mode == "full" and not args.yes:
        raise SystemExit(
            "Refusing to download the full media corpus without --yes. "
            "Use --mode metadata or --mode sidecars for smaller downloads."
        )

    args.local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.local_dir,
        allow_patterns=allow_patterns,
        max_workers=args.max_workers,
    )
    print(f"Downloaded snapshot to: {path}")


if __name__ == "__main__":
    main()
