"""Deterministic per-prefix seeds derived from decoder-available content."""

from __future__ import annotations

import hashlib
import struct
from typing import List, Sequence

import numpy as np
import torch


_HASH_PERSON = b"InfPrefixSeedV1"
_LENGTH = struct.Struct("<Q")
_SCALE_HEADER = struct.Struct("<IQ")
_MAX_SEED = (1 << 63) - 1


def _update_bytes_with_length(hasher, value: bytes) -> None:
    hasher.update(_LENGTH.pack(len(value)))
    hasher.update(value)


def prefix_hash_salt_tag(codec_salt: str) -> str:
    """Return a short stable identifier for result-directory separation."""
    return hashlib.blake2b(
        str(codec_salt).encode("utf-8"), digest_size=4, person=b"InfSaltTagV1"
    ).hexdigest()


def derive_prefix_hash_seeds(
    decoded_indices: Sequence[torch.Tensor],
    prompt: str,
    codec_salt: str = "InfinityCodecPrefixSeedV1",
) -> List[int]:
    """Return one reproducible seed for every decoded scale prefix.

    Only information already available to the decoder is hashed. The evolving
    hash state makes this linear in the total number of decoded token bits.
    """
    hasher = hashlib.blake2b(digest_size=16, person=_HASH_PERSON)
    _update_bytes_with_length(hasher, str(codec_salt).encode("utf-8"))
    _update_bytes_with_length(hasher, str(prompt).encode("utf-8"))

    seeds: List[int] = []
    for scale_index, indices in enumerate(decoded_indices):
        bits = (
            indices.detach()
            .to(device="cpu", dtype=torch.uint8)
            .contiguous()
            .reshape(-1)
            .numpy()
        )
        if bits.size and np.any(bits > 1):
            raise ValueError("prefix hash seed requires binary token indices")
        packed = np.packbits(bits, bitorder="big").tobytes()
        hasher.update(_SCALE_HEADER.pack(int(scale_index), int(bits.size)))
        hasher.update(packed)
        seed = int.from_bytes(hasher.copy().digest()[:8], "little") & _MAX_SEED
        seeds.append(seed)
    return seeds
