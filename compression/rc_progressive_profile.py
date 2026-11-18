"""Shared profile helpers for progressive target-bpp rate control."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Sequence, Tuple


DEFAULT_RATE_HASH_SALT = "InfinityCodecRateControlSeedV1"


def normalize_scale_schedule(scale_schedule: Iterable[Sequence[int]]) -> Tuple[Tuple[int, int, int], ...]:
    return tuple(tuple(int(value) for value in scale) for scale in scale_schedule)


def schedule_id(scale_schedule: Iterable[Sequence[int]]) -> str:
    schedule = normalize_scale_schedule(scale_schedule)
    payload = json.dumps(schedule, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=6).hexdigest()


def salt_tag(salt: str) -> str:
    return hashlib.blake2b(str(salt).encode("utf-8"), digest_size=4).hexdigest()


def canonical_prompt(prompt: str) -> str:
    return " ".join(str(prompt or "").strip().split())


def derive_rate_control_seed(
    *,
    prompt: str,
    target_payload_bits: int,
    ms: str,
    bit_depth: int,
    scale_schedule,
    salt: str = DEFAULT_RATE_HASH_SALT,
    progressive_policy: str = "full_prefix_then_boundary_partial",
) -> int:
    """Derive a deterministic per-image/per-rate seed without stream side info."""

    payload = {
        "salt": str(salt),
        "prompt": canonical_prompt(prompt),
        "target_payload_bits": int(target_payload_bits),
        "ms": str(ms),
        "bit_depth": int(bit_depth),
        "schedule_id": schedule_id(scale_schedule),
        "progressive_policy": str(progressive_policy),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    digest = hashlib.blake2b(encoded, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)
