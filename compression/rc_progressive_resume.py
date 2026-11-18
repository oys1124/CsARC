"""CSV checkpoints for progressive rate-control evaluation."""

from __future__ import annotations

import csv
import fcntl
import json
import math
import os
from pathlib import Path


TEXT_COLUMNS = (
    "image", "ms", "prompt_payload_mode", "per_scale_kept_pos",
    "per_scale_kept_bits", "partial_token_hamming_hist", "seed_policy",
    "hash_seed_tag", "bit_select_order", "cfg_insertion_layer",
)


def result_key(image, ms, target_bpp):
    target = float(target_bpp)
    if not math.isfinite(target) or target <= 0:
        raise ValueError(f"Invalid target_bpp: {target_bpp!r}")
    return str(image), str(ms), target


def acquire_output_lock(out_dir):
    handle = open(Path(out_dir) / ".rate_control_progressive.lock", "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"Another evaluator is writing to {out_dir}") from None
    return handle


def validate_saved_config(path, current):
    path = Path(path)
    if not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    runtime_keys = {
        "out_dir", "caption_json", "data_dir", "target_bpps", "ms_list",
        "max_images", "sample_seed", "resume", "dry_run", "keep_gpu_busy",
        "write_streams", "save_recon", "fsync", "cache_dir", "enable_model_cache",
    }
    for key in (previous.keys() & current.keys()) - runtime_keys:
        if previous[key] != current[key]:
            raise ValueError(
                f"Resume configuration mismatch for {key}: "
                f"saved={previous[key]!r}, requested={current[key]!r}"
            )


def read_completed_keys(csv_path, fields, expected_keys, expected_parameters):
    """Reject corrupt, duplicate or incompatible records before any append."""
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    completed = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != list(fields):
            raise ValueError(f"Resume CSV header does not match this evaluator: {path}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Incomplete CSV record near line {reader.line_num}: {path}")
            key = result_key(row["image"], row["ms"], row["target_bpp"])
            if key not in expected_keys:
                raise ValueError(f"CSV record is outside the requested dataset/rates: {key}")
            if key in completed:
                raise ValueError(f"Duplicate CSV record: {key}")
            for column, expected in expected_parameters[key[1:]].items():
                actual = row[column]
                if isinstance(expected, (int, float)):
                    matches = math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
                else:
                    matches = actual == str(expected)
                if not matches:
                    raise ValueError(
                        f"Resume parameter mismatch for {column} at {key}: "
                        f"CSV={actual!r}, requested={expected!r}"
                    )
            completed.add(key)
    return completed


def open_result_csv(csv_path, fields, resume):
    """Append without rewriting completed records, including a missing final newline."""
    path = Path(csv_path)
    append = bool(resume) and path.exists() and path.stat().st_size > 0
    needs_newline = False
    if append:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) not in (b"\n", b"\r")
    handle = path.open("a" if append else "w", newline="", encoding="utf-8")
    if needs_newline:
        handle.write("\n")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if not append:
        writer.writeheader()
    handle.flush()
    return handle, writer


def write_config(path, values):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def write_summary(csv_path, summary_path):
    """Aggregate the entire checkpoint, including records from previous runs."""
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(csv_path, dtype={column: str for column in TEXT_COLUMNS})
    if frame.empty:
        raise ValueError("Cannot summarize an empty evaluation CSV")
    groups = frame.groupby(["ms", "target_bpp"], as_index=False)
    numeric = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column != "target_bpp"
    ]
    summary = groups[numeric].mean()
    summary["n_images"] = groups["image"].nunique()["image"].to_numpy()
    path = Path(summary_path)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    summary.to_csv(temp, index=False)
    os.replace(temp, path)
    return summary
