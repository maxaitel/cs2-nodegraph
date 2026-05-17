#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import urllib.request
from urllib.parse import quote

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DATASET_ID = "blanchon/opencs2_dataset"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
HF_BASE = f"hf://datasets/{DATASET_ID}"


LABELS = [
    "ace",
    "flick",
    "clutch",
    "1v1",
    "impressive_multikill",
    "other",
]

COLORS = {
    "ace": "#d7263d",
    "flick": "#6f4bd8",
    "clutch": "#f28e2b",
    "1v1": "#2a9d8f",
    "impressive_multikill": "#4e79a7",
    "other": "#8a8f98",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lightweight OpenCS2 play vector graph without downloading videos."
    )
    parser.add_argument("--out-dir", default="outputs", type=Path)
    parser.add_argument("--max-events", default=1200, type=int)
    parser.add_argument("--max-tick-events", default=350, type=int)
    parser.add_argument("--tick-workers", default=8, type=int)
    parser.add_argument("--neighbors", default=7, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--skip-ticks", action="store_true")
    parser.add_argument("--tick-before", default=0.45, type=float)
    parser.add_argument("--tick-after", default=0.10, type=float)
    return parser.parse_args()


def sql_quote(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def hf_path_to_url(path: Any) -> str:
    text = str(path or "")
    match = re.match(r"^hf://datasets/([^@]+)@([^/]+)/(.+)$", text)
    if not match:
        return ""
    repo_id, revision, filename = match.groups()
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{quote(filename, safe='/=')}"


def connect_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    return con


def materialize_remote_tables(con: duckdb.DuckDBPyConnection) -> None:
    print("Loading remote event/index Parquet tables through DuckDB httpfs...")
    con.execute(f"CREATE OR REPLACE TEMP TABLE kills AS SELECT * FROM '{HF_BASE}/events/kills.parquet'")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE round_player AS
        SELECT * FROM '{HF_BASE}/events/round_player.parquet'
        """
    )
    con.execute(f"CREATE OR REPLACE TEMP TABLE rounds AS SELECT * FROM '{HF_BASE}/index/rounds.parquet'")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE pov AS
        SELECT
          media_id,
          match_id,
          map_name,
          round,
          player_slot,
          player_side,
          primary_weapon,
          duration_s,
          media_bytes,
          sidecar_bytes,
          struct_extract(video, 'path') AS video_path,
          ticks_parquet_path
        FROM '{HF_BASE}/index/pov_rounds.parquet'
        """
    )


def fetch_candidates(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    materialize_remote_tables(con)
    print("Building candidate play table from kills, rounds, player stats, and POV metadata...")
    query = """
    WITH player_round_span AS (
      SELECT
        match_id,
        map_name,
        round,
        attacker_player_slot AS player_slot,
        COUNT(*) AS player_round_kills,
        MIN(event_seconds) AS first_player_kill_s,
        MAX(event_seconds) AS last_player_kill_s,
        MAX(event_seconds) - MIN(event_seconds) AS round_kill_span_s
      FROM kills
      GROUP BY match_id, map_name, round, attacker_player_slot
    ),
    kill_windows AS (
      SELECT
        k.kill_id,
        COUNT(k2.kill_id) FILTER (
          WHERE ABS(k2.event_seconds - k.event_seconds) <= 5.0
        ) AS kills_within_10s,
        COUNT(k2.kill_id) FILTER (
          WHERE k2.event_seconds BETWEEN k.event_seconds AND k.event_seconds + 10.0
        ) AS kills_after_10s,
        COUNT(k2.kill_id) FILTER (
          WHERE k2.event_seconds BETWEEN k.event_seconds - 3.0 AND k.event_seconds + 3.0
        ) AS kills_within_6s,
        MAX(k2.event_seconds) FILTER (
          WHERE ABS(k2.event_seconds - k.event_seconds) <= 5.0
        ) - MIN(k2.event_seconds) FILTER (
          WHERE ABS(k2.event_seconds - k.event_seconds) <= 5.0
        ) AS local_kill_span_10s
      FROM kills k
      LEFT JOIN kills k2
        ON k.match_id = k2.match_id
       AND k.map_name = k2.map_name
       AND k.round = k2.round
       AND k.attacker_player_slot = k2.attacker_player_slot
      GROUP BY k.kill_id
    )
    SELECT
      k.kill_id AS play_id,
      k.match_id,
      k.map_name,
      k.map_id,
      k.round,
      k.kill_ordinal,
      k.tick,
      k.event_seconds,
      k.attacker_side,
      k.attacker_side_id,
      k.attacker_player_slot AS player_slot,
      k.victim_side,
      k.victim_player_slot,
      k.weapon,
      k.weapon_id,
      k.weapon_class,
      k.weapon_class_id,
      k.headshot,
      k.distance,
      k.dmg_health,
      k.hitgroup,
      k.noscope,
      k.through_smoke,
      k.penetrated,
      k.wallbang,
      k.attacker_blind,
      k.attacker_kills_before,
      k.attacker_kills_after,
      k.ct_alive_before,
      k.t_alive_before,
      k.ct_alive_after,
      k.t_alive_after,
      k.attacker_alive_before,
      k.opponent_alive_before,
      k.attacker_alive_after,
      k.opponent_alive_after,
      k.time_since_previous_kill_s,
      k.time_to_next_kill_s,
      k.is_opening_kill,
      k.is_trade_within_5s,
      k.is_1v1_before,
      k.is_clutch_context,
      COALESCE(prs.player_round_kills, rp.kills, 0) AS player_round_kills,
      COALESCE(prs.first_player_kill_s, k.event_seconds) AS first_player_kill_s,
      COALESCE(prs.last_player_kill_s, k.event_seconds) AS last_player_kill_s,
      COALESCE(prs.round_kill_span_s, 0.0) AS round_kill_span_s,
      COALESCE(rp.assists, 0) AS player_round_assists,
      COALESCE(rp.headshots, 0) AS player_round_headshots,
      COALESCE(rp.kast, false) AS player_round_kast,
      r.round_duration_s,
      r.winner_side,
      r.reason AS round_reason,
      r.has_bomb_plant,
      r.n_kills AS round_n_kills,
      r.n_headshots AS round_n_headshots,
      r.n_awp_kills AS round_n_awp_kills,
      r.n_smoke_kills AS round_n_smoke_kills,
      r.n_blind_kills AS round_n_blind_kills,
      r.n_noscope_kills AS round_n_noscope_kills,
      r.n_wallbang_kills AS round_n_wallbang_kills,
      r.n_trade_kills_5s AS round_n_trade_kills_5s,
      r.n_1v1_kills AS round_n_1v1_kills,
      r.had_clutch_context,
      r.had_1v1,
      COALESCE(kw.kills_within_10s, 1) AS kills_within_10s,
      COALESCE(kw.kills_after_10s, 1) AS kills_after_10s,
      COALESCE(kw.kills_within_6s, 1) AS kills_within_6s,
      COALESCE(kw.local_kill_span_10s, 0.0) AS local_kill_span_10s,
      p.media_id,
      p.player_side,
      p.primary_weapon,
      p.duration_s,
      p.media_bytes,
      p.sidecar_bytes,
      p.video_path,
      p.ticks_parquet_path
    FROM kills k
    LEFT JOIN player_round_span prs
      ON k.match_id = prs.match_id
     AND k.map_name = prs.map_name
     AND k.round = prs.round
     AND k.attacker_player_slot = prs.player_slot
    LEFT JOIN kill_windows kw
      ON k.kill_id = kw.kill_id
    LEFT JOIN round_player rp
      ON k.match_id = rp.match_id
     AND k.map_name = rp.map_name
     AND k.round = rp.round
     AND k.attacker_player_slot = rp.player_slot
    LEFT JOIN rounds r
      ON k.match_id = r.match_id
     AND k.map_name = r.map_name
     AND k.round = r.round
    INNER JOIN pov p
      ON k.match_id = p.match_id
     AND k.map_name = p.map_name
     AND k.round = p.round
     AND k.attacker_player_slot = p.player_slot
    """
    df = con.sql(query).df()
    print(f"Candidate rows: {len(df):,}")
    return df


def add_base_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bool_cols = [
        "headshot",
        "noscope",
        "through_smoke",
        "wallbang",
        "attacker_blind",
        "is_opening_kill",
        "is_trade_within_5s",
        "is_1v1_before",
        "is_clutch_context",
        "player_round_kast",
        "has_bomb_plant",
        "had_clutch_context",
        "had_1v1",
    ]
    for col in bool_cols:
        if col in df:
            df[col] = df[col].fillna(False).astype(bool)

    numeric_defaults = {
        "distance": 0.0,
        "dmg_health": 0,
        "penetrated": 0,
        "player_round_kills": 0,
        "round_kill_span_s": 0.0,
        "kills_within_10s": 1,
        "kills_after_10s": 1,
        "kills_within_6s": 1,
        "local_kill_span_10s": 0.0,
        "time_since_previous_kill_s": 999.0,
        "time_to_next_kill_s": 999.0,
    }
    for col, default in numeric_defaults.items():
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    quality = (
        df["headshot"]
        | df["wallbang"]
        | df["through_smoke"]
        | df["noscope"]
        | (df["penetrated"].fillna(0) > 0)
    )
    pace = (df["kills_within_10s"] >= 3) | (df["kills_within_6s"] >= 2)

    df["is_ace"] = df["player_round_kills"] >= 5
    df["is_1v1"] = df["is_1v1_before"]
    df["is_clutch"] = df["is_clutch_context"]
    df["is_plain_multikill"] = df["player_round_kills"] >= 2
    df["is_impressive_multikill"] = (
        (df["player_round_kills"] >= 4)
        | ((df["player_round_kills"] >= 3) & pace)
        | ((df["player_round_kills"] >= 3) & quality & (df["round_kill_span_s"] <= 20.0))
        | ((df["player_round_kills"] >= 3) & df["is_clutch"])
    )
    df["is_flick"] = False
    df["tick_feature_computed"] = False
    df["view_snap_score"] = 0.0
    df["flick_score_percentile"] = np.nan

    weapon_text = (df["weapon"].fillna("") + " " + df["weapon_class"].fillna("")).str.lower()
    precision_weapon = weapon_text.str.contains("awp|ssg|sniper|deagle|revolver|rifle", regex=True)
    df["interest_score"] = (
        3.5 * df["is_ace"].astype(float)
        + 2.4 * df["is_clutch"].astype(float)
        + 2.0 * df["is_1v1"].astype(float)
        + 2.2 * df["is_impressive_multikill"].astype(float)
        + 0.7 * df["headshot"].astype(float)
        + 0.9 * df["wallbang"].astype(float)
        + 0.8 * df["through_smoke"].astype(float)
        + 1.0 * df["noscope"].astype(float)
        + 0.4 * precision_weapon.astype(float)
        + 0.35 * np.clip(df["kills_within_10s"].astype(float) - 1.0, 0.0, 5.0)
        + 0.15 * np.log1p(df["distance"].astype(float).clip(lower=0.0))
    )
    return df


def select_tick_candidates(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0:
        return df.head(0)

    pool = df[df["ticks_parquet_path"].notna()].copy()
    if pool.empty:
        return pool

    weapon_text = (pool["weapon"].fillna("") + " " + pool["weapon_class"].fillna("")).str.lower()
    precision_weapon = weapon_text.str.contains("awp|ssg|sniper|deagle|revolver|rifle", regex=True)
    pool["_flick_proxy"] = (
        2.5 * pool["headshot"].astype(float)
        + 1.5 * precision_weapon.astype(float)
        + 1.0 * pool["is_1v1"].astype(float)
        + 0.8 * pool["is_clutch"].astype(float)
        + 0.6 * pool["noscope"].astype(float)
        + 0.35 * np.log1p(pool["distance"].astype(float).clip(lower=0.0))
    )

    selected: list[Any] = []

    def add_indices(indices: list[Any]) -> None:
        seen = set(selected)
        for idx in indices:
            if idx not in seen and len(selected) < limit:
                selected.append(idx)
                seen.add(idx)

    top_n = min(len(pool), max(1, int(limit * 0.72)))
    add_indices(pool.sort_values("_flick_proxy", ascending=False).head(top_n).index.tolist())

    per_label = max(3, int(limit * 0.07))
    rng_seed = seed
    for col in ["is_ace", "is_clutch", "is_1v1", "is_impressive_multikill"]:
        if len(selected) >= limit:
            break
        label_pool = pool[pool[col] & ~pool.index.isin(selected)]
        n = min(per_label, len(label_pool), limit - len(selected))
        if n > 0:
            add_indices(label_pool.sample(n=n, random_state=rng_seed).index.tolist())
            rng_seed += 1

    if len(selected) < limit:
        remainder = pool[~pool.index.isin(selected)].sort_values("_flick_proxy", ascending=False)
        add_indices(remainder.head(limit - len(selected)).index.tolist())

    return pool.loc[selected].drop(columns=["_flick_proxy"], errors="ignore")


def read_tick_feature_row(record: dict[str, Any], before: float, after: float) -> dict[str, Any]:
    play_id = record["play_id"]
    event_s = float(record["event_seconds"])
    start_s = max(0.0, event_s - before)
    stop_s = event_s + after
    url = hf_path_to_url(record["ticks_parquet_path"])
    base = {
        "play_id": play_id,
        "tick_window_n": 0,
        "pre_angle_sum": 0.0,
        "pre_angle_max": 0.0,
        "pre_angle_mean": 0.0,
        "pre250_angle_sum": 0.0,
        "pre250_angle_max": 0.0,
        "pre250_yaw_sum": 0.0,
        "pre250_pitch_sum": 0.0,
        "post_angle_sum": 0.0,
        "avg_speed_xy": 0.0,
        "max_speed_xy": 0.0,
        "tick_feature_computed": False,
        "tick_error": "",
    }
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            payload = response.read()
        table = pq.read_table(
            pa.BufferReader(payload),
            columns=["t", "delta_yaw", "delta_pitch", "velocity_x", "velocity_y"],
        )

        def arr(name: str) -> np.ndarray:
            return np.nan_to_num(
                np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        t = arr("t")
        window = (t >= start_s) & (t <= stop_s)
        if not np.any(window):
            return base

        t = t[window]
        delta_yaw = arr("delta_yaw")[window]
        delta_pitch = arr("delta_pitch")[window]
        velocity_x = arr("velocity_x")[window]
        velocity_y = arr("velocity_y")[window]

        angle_delta = np.sqrt(delta_yaw * delta_yaw + delta_pitch * delta_pitch)
        yaw_delta_abs = np.abs(delta_yaw)
        pitch_delta_abs = np.abs(delta_pitch)
        speed_xy = np.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)

        pre = t <= event_s
        pre250 = (t >= event_s - 0.250) & (t <= event_s)
        post = t > event_s

        def sum_mask(values: np.ndarray, mask: np.ndarray) -> float:
            return float(values[mask].sum()) if np.any(mask) else 0.0

        def max_mask(values: np.ndarray, mask: np.ndarray) -> float:
            return float(values[mask].max()) if np.any(mask) else 0.0

        def mean_mask(values: np.ndarray, mask: np.ndarray) -> float:
            return float(values[mask].mean()) if np.any(mask) else 0.0

        base.update(
            {
                "tick_window_n": int(window.sum()),
                "pre_angle_sum": sum_mask(angle_delta, pre),
                "pre_angle_max": max_mask(angle_delta, pre),
                "pre_angle_mean": mean_mask(angle_delta, pre),
                "pre250_angle_sum": sum_mask(angle_delta, pre250),
                "pre250_angle_max": max_mask(angle_delta, pre250),
                "pre250_yaw_sum": sum_mask(yaw_delta_abs, pre250),
                "pre250_pitch_sum": sum_mask(pitch_delta_abs, pre250),
                "post_angle_sum": sum_mask(angle_delta, post),
                "avg_speed_xy": float(speed_xy.mean()) if len(speed_xy) else 0.0,
                "max_speed_xy": float(speed_xy.max()) if len(speed_xy) else 0.0,
                "tick_feature_computed": True,
            }
        )
        return base
    except Exception as exc:
        base["tick_error"] = str(exc)[:240]
        return base


def fetch_tick_features(
    rows: pd.DataFrame,
    before: float,
    after: float,
    workers: int,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()

    print(f"Reading {len(rows):,} bounded tick sidecars for flick inference...")
    results: list[dict[str, Any]] = []
    started = time.time()
    records = rows.to_dict("records")
    max_workers = max(1, min(workers, len(records)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(read_tick_feature_row, record, before, after)
            for record in records
        ]
        for n, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            if n == 1 or n % 25 == 0 or n == len(rows):
                elapsed = time.time() - started
                print(f"  tick sidecars {n:,}/{len(rows):,} ({elapsed:.1f}s)")

    return pd.DataFrame(results)


def add_flick_labels(df: pd.DataFrame, tick_features: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    df = df.copy()
    if not tick_features.empty:
        df = df.merge(tick_features, on="play_id", how="left", suffixes=("", "_tick"))
        if "tick_feature_computed_tick" in df:
            df["tick_feature_computed"] = df["tick_feature_computed_tick"].fillna(False).astype(bool)
            df = df.drop(columns=["tick_feature_computed_tick"])

    tick_cols = [
        "tick_window_n",
        "pre_angle_sum",
        "pre_angle_max",
        "pre_angle_mean",
        "pre250_angle_sum",
        "pre250_angle_max",
        "pre250_yaw_sum",
        "pre250_pitch_sum",
        "post_angle_sum",
        "avg_speed_xy",
        "max_speed_xy",
    ]
    for col in tick_cols:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "tick_error" not in df:
        df["tick_error"] = ""
    df["tick_error"] = df["tick_error"].fillna("")
    df["tick_feature_computed"] = df["tick_feature_computed"].fillna(False).astype(bool)

    distance_bonus = 0.025 * np.log1p(df["distance"].astype(float).clip(lower=0.0))
    precision_bonus = (
        0.35 * df["headshot"].astype(float)
        + 0.20 * df["is_1v1"].astype(float)
        + 0.10 * df["noscope"].astype(float)
    )
    df["view_snap_score"] = (
        df["pre250_angle_sum"]
        + 2.4 * df["pre250_angle_max"]
        + 0.20 * df["pre_angle_sum"]
        + distance_bonus
        + precision_bonus
    )

    computed = df["tick_feature_computed"] & np.isfinite(df["view_snap_score"])
    if computed.sum() < 10:
        df["is_flick"] = False
        return df, None

    scores = df.loc[computed, "view_snap_score"]
    threshold = float(scores.quantile(0.88))
    df.loc[computed, "flick_score_percentile"] = scores.rank(pct=True)

    eligible = computed & (df["view_snap_score"] >= threshold)
    target_min = min(45, max(10, int(math.ceil(computed.sum() * 0.10))))
    if int(eligible.sum()) < target_min:
        top_idx = df.loc[computed].nlargest(target_min, "view_snap_score").index
        eligible = df.index.isin(top_idx)

    df["is_flick"] = eligible
    return df, threshold


def assign_tags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def tags_for(row: pd.Series) -> list[str]:
        tags: list[str] = []
        if row.get("is_ace", False):
            tags.append("ace")
        if row.get("is_flick", False):
            tags.append("flick")
        if row.get("is_clutch", False):
            tags.append("clutch")
        if row.get("is_1v1", False):
            tags.append("1v1")
        if row.get("is_impressive_multikill", False):
            tags.append("impressive_multikill")
        return tags or ["other"]

    tag_lists = df.apply(tags_for, axis=1)
    df["tags"] = tag_lists.apply(lambda values: ";".join(values))

    def primary_for(row: pd.Series) -> str:
        if row.get("is_ace", False):
            return "ace"
        if row.get("is_flick", False):
            return "flick"
        if row.get("is_impressive_multikill", False) and row.get("player_round_kills", 0) >= 4:
            return "impressive_multikill"
        if row.get("is_1v1", False):
            return "1v1"
        if row.get("is_clutch", False):
            return "clutch"
        if row.get("is_impressive_multikill", False):
            return "impressive_multikill"
        return "other"

    df["primary_label"] = df.apply(primary_for, axis=1)
    return df


def stratified_sample(df: pd.DataFrame, max_events: int, seed: int) -> pd.DataFrame:
    if max_events <= 0 or len(df) <= max_events:
        return df.sort_values("interest_score", ascending=False).reset_index(drop=True)

    selected: list[Any] = []
    quotas = [
        ("ace", "is_ace", 0.18),
        ("flick", "is_flick", 0.16),
        ("clutch", "is_clutch", 0.18),
        ("1v1", "is_1v1", 0.16),
        ("impressive_multikill", "is_impressive_multikill", 0.18),
        ("other", None, 0.14),
    ]

    def add_indices(indices: list[Any]) -> None:
        seen = set(selected)
        for idx in indices:
            if idx not in seen and len(selected) < max_events:
                selected.append(idx)
                seen.add(idx)

    for n, (label, col, frac) in enumerate(quotas):
        target = int(round(max_events * frac))
        if label == "other":
            pool = df[(df["primary_label"] == "other") & ~df.index.isin(selected)].copy()
        else:
            pool = df[(df["primary_label"] == label) & ~df.index.isin(selected)].copy()
            if pool.empty and col is not None:
                pool = df[df[col] & ~df.index.isin(selected)].copy()
        if pool.empty:
            continue
        chosen = pool.sort_values("interest_score", ascending=False).head(target)
        if len(chosen) < target:
            remainder = pool.drop(chosen.index, errors="ignore")
            extra_n = min(target - len(chosen), len(remainder))
            if extra_n > 0:
                chosen = pd.concat(
                    [chosen, remainder.sample(n=extra_n, random_state=seed + n)],
                    axis=0,
                )
        add_indices(chosen.index.tolist())

    while len(selected) < max_events:
        added = False
        counts = df.loc[selected, "primary_label"].value_counts().to_dict() if selected else {}
        for label in sorted(LABELS, key=lambda value: counts.get(value, 0)):
            pool = df[(df["primary_label"] == label) & ~df.index.isin(selected)]
            if pool.empty:
                continue
            add_indices(pool.sort_values("interest_score", ascending=False).head(1).index.tolist())
            added = True
            if len(selected) >= max_events:
                break
        if not added:
            break

    remaining = max_events - len(selected)
    if remaining > 0:
        pool = df[~df.index.isin(selected)].copy()
        n = min(remaining, len(pool))
        if n > 0:
            add_indices(pool.sample(n=n, random_state=seed + 99).index.tolist())

    sample = df.loc[selected].copy()
    return sample.sort_values(["primary_label", "interest_score"], ascending=[True, False]).reset_index(drop=True)


def vectorize(sample: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = sample.copy()
    safe_duration = pd.to_numeric(df["duration_s"], errors="coerce").replace(0, np.nan)
    df["event_phase"] = (
        pd.to_numeric(df["event_seconds"], errors="coerce") / safe_duration
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.5)
    df["distance_log"] = np.log1p(pd.to_numeric(df["distance"], errors="coerce").fillna(0.0).clip(lower=0.0))
    df["time_since_previous_kill_log"] = np.log1p(
        pd.to_numeric(df["time_since_previous_kill_s"], errors="coerce").fillna(999.0).clip(0.0, 60.0)
    )
    df["time_to_next_kill_log"] = np.log1p(
        pd.to_numeric(df["time_to_next_kill_s"], errors="coerce").fillna(999.0).clip(0.0, 60.0)
    )
    df["media_mb"] = pd.to_numeric(df["media_bytes"], errors="coerce").fillna(0.0) / 1_000_000.0
    df["sidecar_kb"] = pd.to_numeric(df["sidecar_bytes"], errors="coerce").fillna(0.0) / 1_000.0

    bool_features = [
        "headshot",
        "noscope",
        "through_smoke",
        "wallbang",
        "attacker_blind",
        "is_opening_kill",
        "is_trade_within_5s",
        "has_bomb_plant",
        "had_clutch_context",
        "had_1v1",
        "tick_feature_computed",
    ]
    numeric_features = [
        "event_phase",
        "event_seconds",
        "distance_log",
        "dmg_health",
        "penetrated",
        "attacker_kills_before",
        "attacker_kills_after",
        "ct_alive_before",
        "t_alive_before",
        "ct_alive_after",
        "t_alive_after",
        "attacker_alive_before",
        "opponent_alive_before",
        "attacker_alive_after",
        "opponent_alive_after",
        "time_since_previous_kill_log",
        "time_to_next_kill_log",
        "player_round_kills",
        "player_round_assists",
        "player_round_headshots",
        "round_kill_span_s",
        "round_n_kills",
        "round_n_headshots",
        "round_n_awp_kills",
        "round_n_smoke_kills",
        "round_n_blind_kills",
        "round_n_noscope_kills",
        "round_n_wallbang_kills",
        "round_n_trade_kills_5s",
        "round_n_1v1_kills",
        "kills_within_10s",
        "kills_after_10s",
        "kills_within_6s",
        "local_kill_span_10s",
        "tick_window_n",
        "pre_angle_sum",
        "pre_angle_max",
        "pre_angle_mean",
        "pre250_angle_sum",
        "pre250_angle_max",
        "pre250_yaw_sum",
        "pre250_pitch_sum",
        "post_angle_sum",
        "avg_speed_xy",
        "max_speed_xy",
        "view_snap_score",
        "media_mb",
        "sidecar_kb",
        "interest_score",
    ]
    categorical_features = ["map_name", "weapon_class", "weapon", "attacker_side", "round_reason"]

    feature_frames: list[pd.DataFrame] = []
    for col in bool_features:
        if col in df:
            feature_frames.append(df[[col]].fillna(False).astype(float))
    existing_numeric = [col for col in numeric_features if col in df]
    feature_frames.append(
        df[existing_numeric]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    feature_frames.append(pd.get_dummies(df[categorical_features].fillna("unknown"), prefix=categorical_features))

    features = pd.concat(feature_frames, axis=1)
    feature_names = features.columns.tolist()
    matrix = features.to_numpy(dtype=float)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-9] = 1.0
    matrix_z = (matrix - mean) / std

    centered = matrix_z - matrix_z.mean(axis=0, keepdims=True)
    if len(df) >= 2 and centered.shape[1] >= 2:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T
    else:
        coords = np.zeros((len(df), 2))
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))

    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    return df, matrix_z, feature_names


def build_edges(matrix_z: np.ndarray, neighbors: int) -> pd.DataFrame:
    n = matrix_z.shape[0]
    if n <= 1:
        return pd.DataFrame(columns=["source", "target", "similarity", "weight"])

    k = max(1, min(neighbors, n - 1))
    norms = np.linalg.norm(matrix_z, axis=1)
    norms[norms < 1e-9] = 1.0
    normalized = matrix_z / norms[:, None]
    sim = normalized @ normalized.T
    np.fill_diagonal(sim, -np.inf)

    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        idx = np.argpartition(-sim[i], kth=k - 1)[:k]
        idx = idx[np.argsort(-sim[i, idx])]
        for j in idx:
            if not np.isfinite(sim[i, j]):
                continue
            a, b = sorted((int(i), int(j)))
            edges[(a, b)] = max(edges.get((a, b), -1.0), float(sim[i, j]))

    rows = [
        {
            "source": a,
            "target": b,
            "similarity": score,
            "weight": max(0.05, min(1.0, (score + 1.0) / 2.0)),
        }
        for (a, b), score in edges.items()
    ]
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)


def normalize_for_graph(values: pd.Series, lo: float, hi: float) -> list[float]:
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(arr) == 0:
        return []
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if abs(max_v - min_v) < 1e-9:
        return [float((lo + hi) / 2.0)] * len(arr)
    return (lo + (arr - min_v) * (hi - lo) / (max_v - min_v)).tolist()


def graph_nodes(df: pd.DataFrame) -> list[dict[str, Any]]:
    gx = normalize_for_graph(df["x"], 70, 1130)
    gy = normalize_for_graph(df["y"], 70, 730)
    nodes: list[dict[str, Any]] = []
    for i, row in df.reset_index(drop=True).iterrows():
        tags = str(row.get("tags", "other")).split(";")
        nodes.append(
            {
                "idx": int(i),
                "id": str(row["play_id"]),
                "label": str(row["primary_label"]),
                "tags": tags,
                "x": gx[i],
                "y": gy[i],
                "radius": float(3.2 + min(4.8, max(0.0, row.get("interest_score", 0.0)) * 0.35)),
                "map": str(row.get("map_name", "")),
                "round": int(row.get("round", 0)),
                "player": int(row.get("player_slot", 0)),
                "weapon": str(row.get("weapon", "")),
                "weapon_class": str(row.get("weapon_class", "")),
                "event_seconds": round(float(row.get("event_seconds", 0.0)), 3),
                "distance": round(float(row.get("distance", 0.0)), 2),
                "player_round_kills": int(row.get("player_round_kills", 0)),
                "kills_within_10s": int(row.get("kills_within_10s", 0)),
                "view_snap_score": round(float(row.get("view_snap_score", 0.0)), 3),
                "flick_percentile": (
                    None
                    if pd.isna(row.get("flick_score_percentile", np.nan))
                    else round(float(row.get("flick_score_percentile")), 3)
                ),
                "tick_features": bool(row.get("tick_feature_computed", False)),
                "video_url": hf_path_to_url(row.get("video_path")),
                "tick_path": str(row.get("ticks_parquet_path", "")),
            }
        )
    return nodes


def write_html(df: pd.DataFrame, edges: pd.DataFrame, summary: dict[str, Any], out_path: Path) -> None:
    nodes = graph_nodes(df)
    edge_rows = [
        {
            "source": int(row["source"]),
            "target": int(row["target"]),
            "weight": round(float(row["weight"]), 4),
            "similarity": round(float(row["similarity"]), 4),
        }
        for _, row in edges.iterrows()
    ]
    payload = {
        "nodes": nodes,
        "edges": edge_rows,
        "labels": LABELS,
        "colors": COLORS,
        "summary": summary,
    }
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenCS2 Play Graph</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1d1f24;
      --muted: #667085;
      --line: #d7d9de;
      --accent: #264653;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.4 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      letter-spacing: 0;
      font-weight: 700;
    }}
    .stats {{
      display: flex;
      gap: 16px;
      color: var(--muted);
      font-size: 13px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(560px, 1fr) 340px;
      min-height: calc(100vh - 65px);
    }}
    .graph-wrap {{
      min-width: 0;
      padding: 16px;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .search {{
      min-width: 230px;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: var(--ink);
    }}
    .check {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      height: 34px;
      padding: 0 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      white-space: nowrap;
      color: #344054;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }}
    svg {{
      width: 100%;
      height: calc(100vh - 135px);
      min-height: 560px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fcfcfb;
      cursor: grab;
      touch-action: none;
    }}
    svg.dragging {{ cursor: grabbing; }}
    .edge {{
      stroke: #98a2b3;
      stroke-linecap: round;
    }}
    .node {{
      stroke: rgba(255, 255, 255, 0.92);
      stroke-width: 1.3;
      cursor: pointer;
    }}
    .node.dim, .edge.dim {{ opacity: 0.06; }}
    aside {{
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      min-width: 0;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
      background: #fff;
    }}
    .card h2 {{
      margin: 0 0 8px;
      font-size: 15px;
      letter-spacing: 0;
    }}
    .kv {{
      display: grid;
      grid-template-columns: 116px 1fr;
      gap: 6px 10px;
      font-size: 13px;
    }}
    .kv div:nth-child(odd) {{ color: var(--muted); }}
    .tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0 12px;
    }}
    .tag {{
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      color: #fff;
      font-size: 12px;
      font-weight: 650;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 960px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-left: 0; border-top: 1px solid var(--line); }}
      svg {{ height: 68vh; min-height: 460px; }}
      header {{ align-items: flex-start; flex-direction: column; gap: 8px; }}
      .stats {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>OpenCS2 Play Graph</h1>
    <div class="stats" id="stats"></div>
  </header>
  <main>
    <section class="graph-wrap">
      <div class="toolbar">
        <input class="search" id="search" placeholder="Search id, map, weapon, tag">
        <span id="filters"></span>
      </div>
      <svg id="graph" viewBox="0 0 1200 800" role="img" aria-label="OpenCS2 nearest-neighbor play graph">
        <g id="viewport">
          <g id="edges"></g>
          <g id="nodes"></g>
        </g>
      </svg>
    </section>
    <aside>
      <div class="card">
        <h2>Selection</h2>
        <div id="selection" class="muted">Select a node.</div>
      </div>
      <div class="card">
        <h2>Run</h2>
        <div id="run" class="kv"></div>
      </div>
    </aside>
  </main>
  <script id="graph-data" type="application/json">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById("graph-data").textContent);
    const nodes = data.nodes;
    const edges = data.edges;
    const colors = data.colors;
    const svg = document.getElementById("graph");
    const viewport = document.getElementById("viewport");
    const edgeLayer = document.getElementById("edges");
    const nodeLayer = document.getElementById("nodes");
    const search = document.getElementById("search");
    const filters = document.getElementById("filters");
    const selection = document.getElementById("selection");
    const stats = document.getElementById("stats");
    const run = document.getElementById("run");
    const nodeByIndex = new Map(nodes.map(n => [n.idx, n]));

    function esc(value) {{
      return String(value ?? "").replace(/[&<>"']/g, ch => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }}[ch]));
    }}

    function labelName(label) {{
      return label === "impressive_multikill" ? "impressive" : label;
    }}

    function drawFilters() {{
      filters.innerHTML = data.labels.map(label => `
        <label class="check">
          <input type="checkbox" data-label="${{esc(label)}}" checked>
          <span class="swatch" style="background:${{colors[label]}}"></span>
          ${{labelName(label)}}
        </label>
      `).join("");
      filters.querySelectorAll("input").forEach(input => input.addEventListener("change", render));
    }}

    function activeLabels() {{
      return new Set([...filters.querySelectorAll("input:checked")].map(input => input.dataset.label));
    }}

    function matchesQuery(node, query) {{
      if (!query) return true;
      const haystack = [
        node.id, node.label, node.tags.join(" "), node.map, node.weapon, node.weapon_class,
        String(node.round), String(node.player)
      ].join(" ").toLowerCase();
      return haystack.includes(query);
    }}

    function visibleNodeSet() {{
      const labels = activeLabels();
      const query = search.value.trim().toLowerCase();
      return new Set(nodes.filter(node => labels.has(node.label) && matchesQuery(node, query)).map(node => node.idx));
    }}

    function showNode(node) {{
      const tagHtml = node.tags.map(tag => `<span class="tag" style="background:${{colors[tag] || colors.other}}">${{esc(labelName(tag))}}</span>`).join("");
      const video = node.video_url ? `<a href="${{esc(node.video_url)}}" target="_blank" rel="noreferrer">open POV video</a>` : `<span class="muted">none</span>`;
      selection.innerHTML = `
        <div><strong>${{esc(node.id)}}</strong></div>
        <div class="tags">${{tagHtml}}</div>
        <div class="kv">
          <div>Map</div><div>${{esc(node.map)}}</div>
          <div>Round</div><div>${{node.round}}</div>
          <div>Player</div><div>${{node.player}}</div>
          <div>Weapon</div><div>${{esc(node.weapon)}}</div>
          <div>Time</div><div>${{node.event_seconds}}s</div>
          <div>Distance</div><div>${{node.distance}}</div>
          <div>Round kills</div><div>${{node.player_round_kills}}</div>
          <div>10s kills</div><div>${{node.kills_within_10s}}</div>
          <div>Flick score</div><div>${{node.view_snap_score}}</div>
          <div>Tick features</div><div>${{node.tick_features ? "yes" : "no"}}</div>
          <div>Video</div><div>${{video}}</div>
        </div>
      `;
    }}

    function render() {{
      const visible = visibleNodeSet();
      edgeLayer.textContent = "";
      nodeLayer.textContent = "";

      for (const edge of edges) {{
        if (!visible.has(edge.source) || !visible.has(edge.target)) continue;
        const a = nodeByIndex.get(edge.source);
        const b = nodeByIndex.get(edge.target);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("class", "edge");
        line.setAttribute("x1", a.x);
        line.setAttribute("y1", a.y);
        line.setAttribute("x2", b.x);
        line.setAttribute("y2", b.y);
        line.setAttribute("stroke-width", 0.4 + 1.7 * edge.weight);
        line.setAttribute("opacity", 0.08 + 0.30 * edge.weight);
        edgeLayer.appendChild(line);
      }}

      for (const node of nodes) {{
        if (!visible.has(node.idx)) continue;
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("class", "node");
        circle.setAttribute("cx", node.x);
        circle.setAttribute("cy", node.y);
        circle.setAttribute("r", node.radius);
        circle.setAttribute("fill", colors[node.label] || colors.other);
        circle.addEventListener("mouseenter", () => showNode(node));
        circle.addEventListener("click", () => showNode(node));
        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        title.textContent = `${{node.label}} | ${{node.map}} r${{node.round}} p${{node.player}} | ${{node.weapon}}`;
        circle.appendChild(title);
        nodeLayer.appendChild(circle);
      }}
      stats.innerHTML = [
        `${{visible.size.toLocaleString()}} shown`,
        `${{nodes.length.toLocaleString()}} plays`,
        `${{edges.length.toLocaleString()}} edges`,
        `${{data.summary.tick_features_computed.toLocaleString()}} tick-featured`
      ].map(text => `<span>${{text}}</span>`).join("");
    }}

    function drawRun() {{
      const counts = Object.entries(data.summary.primary_label_counts)
        .map(([label, count]) => `${{labelName(label)}}: ${{count}}`).join("<br>");
      run.innerHTML = `
        <div>Dataset</div><div><a href="${{esc(data.summary.dataset_url)}}" target="_blank" rel="noreferrer">Hugging Face</a></div>
        <div>Generated</div><div>${{esc(data.summary.generated_at)}}</div>
        <div>Rows scanned</div><div>${{data.summary.candidate_rows.toLocaleString()}}</div>
        <div>Features</div><div>${{data.summary.feature_count}}</div>
        <div>Neighbors</div><div>${{data.summary.neighbors}}</div>
        <div>Counts</div><div>${{counts}}</div>
      `;
    }}

    let scale = 1;
    let panX = 0;
    let panY = 0;
    let dragging = false;
    let last = null;

    function applyTransform() {{
      viewport.setAttribute("transform", `translate(${{panX}} ${{panY}}) scale(${{scale}})`);
    }}

    svg.addEventListener("wheel", event => {{
      event.preventDefault();
      const next = Math.max(0.45, Math.min(4.0, scale * (event.deltaY > 0 ? 0.92 : 1.08)));
      scale = next;
      applyTransform();
    }}, {{ passive: false }});

    svg.addEventListener("pointerdown", event => {{
      dragging = true;
      last = {{ x: event.clientX, y: event.clientY }};
      svg.classList.add("dragging");
      svg.setPointerCapture(event.pointerId);
    }});

    svg.addEventListener("pointermove", event => {{
      if (!dragging || !last) return;
      panX += (event.clientX - last.x) / scale;
      panY += (event.clientY - last.y) / scale;
      last = {{ x: event.clientX, y: event.clientY }};
      applyTransform();
    }});

    svg.addEventListener("pointerup", event => {{
      dragging = false;
      last = null;
      svg.classList.remove("dragging");
      svg.releasePointerCapture(event.pointerId);
    }});

    search.addEventListener("input", render);
    drawFilters();
    drawRun();
    render();
    if (nodes.length) showNode(nodes[0]);
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def write_outputs(
    df: pd.DataFrame,
    edges: pd.DataFrame,
    feature_names: list[str],
    flick_threshold: float | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vector_path = args.out_dir / "opencs2_play_vectors.csv"
    edge_path = args.out_dir / "opencs2_play_edges.csv"
    html_path = args.out_dir / "opencs2_play_graph.html"
    summary_path = args.out_dir / "opencs2_play_summary.json"

    df = df.copy()
    df["video_url"] = df["video_path"].apply(hf_path_to_url)
    df.to_csv(vector_path, index=False)
    edges.to_csv(edge_path, index=False)

    primary_counts = {label: int((df["primary_label"] == label).sum()) for label in LABELS}
    tag_counts = {
        "ace": int(df["is_ace"].sum()),
        "flick": int(df["is_flick"].sum()),
        "clutch": int(df["is_clutch"].sum()),
        "1v1": int(df["is_1v1"].sum()),
        "impressive_multikill": int(df["is_impressive_multikill"].sum()),
        "plain_multikill": int(df["is_plain_multikill"].sum()),
    }
    summary = {
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate_rows": int(args.candidate_rows),
        "sampled_rows": int(len(df)),
        "edge_rows": int(len(edges)),
        "neighbors": int(args.neighbors),
        "max_events": int(args.max_events),
        "max_tick_events": int(0 if args.skip_ticks else args.max_tick_events),
        "tick_features_computed": int(df["tick_feature_computed"].sum()),
        "flick_threshold": flick_threshold,
        "primary_label_counts": primary_counts,
        "tag_counts": tag_counts,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "outputs": {
            "vectors_csv": str(vector_path),
            "edges_csv": str(edge_path),
            "graph_html": str(html_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_html(df, edges, summary, html_path)
    return summary


def main() -> None:
    args = parse_args()
    started = time.time()
    con = connect_duckdb()

    candidates = fetch_candidates(con)
    args.candidate_rows = len(candidates)
    labeled = add_base_labels(candidates)

    tick_features = pd.DataFrame()
    flick_threshold = None
    if not args.skip_ticks and args.max_tick_events > 0:
        tick_rows = select_tick_candidates(labeled, args.max_tick_events, args.seed)
        tick_features = fetch_tick_features(
            tick_rows,
            before=args.tick_before,
            after=args.tick_after,
            workers=args.tick_workers,
        )
    else:
        print("Skipping tick sidecars; flick labels will be empty.")

    labeled, flick_threshold = add_flick_labels(labeled, tick_features)
    labeled = assign_tags(labeled)
    sample = stratified_sample(labeled, args.max_events, args.seed)
    sample, matrix_z, feature_names = vectorize(sample)
    edges = build_edges(matrix_z, args.neighbors)
    summary = write_outputs(sample, edges, feature_names, flick_threshold, args)

    elapsed = time.time() - started
    print("\nDone.")
    print(f"  sampled plays: {summary['sampled_rows']:,}")
    print(f"  graph edges: {summary['edge_rows']:,}")
    print(f"  tick-featured plays: {summary['tick_features_computed']:,}")
    print(f"  graph: {summary['outputs']['graph_html']}")
    print(f"  elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
