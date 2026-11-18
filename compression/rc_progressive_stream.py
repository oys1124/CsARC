"""HPRC stream packing and parsing.

The caption and target-payload budget are stored in the stream. Model, image
profile, bit-selection strategy and seed policy are configured by the caller.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from typing import Optional


TARGET_PAYLOAD_BITS = struct.Struct("<I")
PROMPT_LENGTH = struct.Struct("<H")
PROMPT_MODE_ZLIB = 0
PROMPT_MODE_RAW_UTF8 = 1
PROMPT_MODE_MASK = 0x8000
PROMPT_LENGTH_MASK = 0x7FFF


@dataclass
class ProgressiveStreamContents:
    prompt: str
    payload: bytes
    prompt_bytes: int
    file_bytes: int
    prompt_mode: str = ""
    target_payload_bits: Optional[int] = None
    target_header_bytes: int = 0


def write_payload_stream(path: str, payload: bytes, *, fsync: bool = False) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = bytes(payload)
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    return len(payload)


def read_payload_stream(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def prompt_mode_name(mode: int) -> str:
    mode = int(mode)
    if mode == PROMPT_MODE_ZLIB:
        return "zlib"
    if mode == PROMPT_MODE_RAW_UTF8:
        return "raw_utf8"
    raise ValueError(f"unsupported progressive prompt mode={mode}")


def select_progressive_prompt_payload(prompt: str):
    raw_payload = str(prompt).encode("utf-8")
    zlib_payload = zlib.compress(raw_payload)
    candidates = []
    if len(zlib_payload) <= PROMPT_LENGTH_MASK:
        candidates.append((PROMPT_MODE_ZLIB, zlib_payload))
    if len(raw_payload) <= PROMPT_LENGTH_MASK:
        candidates.append((PROMPT_MODE_RAW_UTF8, raw_payload))
    if not candidates:
        raise ValueError("progressive prompt payload exceeds uint15 length")
    return min(candidates, key=lambda item: len(item[1]))


def pack_progressive_stream(*, prompt: str, payload: bytes) -> bytes:
    prompt_mode, prompt_payload = select_progressive_prompt_payload(prompt)
    descriptor = len(prompt_payload)
    if int(prompt_mode) == PROMPT_MODE_RAW_UTF8:
        descriptor |= PROMPT_MODE_MASK
    return PROMPT_LENGTH.pack(descriptor) + prompt_payload + bytes(payload)


def pack_progressive_target_stream(*, prompt: str, payload: bytes, target_payload_bits: int) -> bytes:
    target_payload_bits = int(target_payload_bits)
    if target_payload_bits < 0 or target_payload_bits > 0xFFFFFFFF:
        raise ValueError(f"target_payload_bits out of uint32 range: {target_payload_bits}")
    return TARGET_PAYLOAD_BITS.pack(target_payload_bits) + pack_progressive_stream(
        prompt=prompt,
        payload=payload,
    )


def unpack_progressive_stream(data: bytes) -> ProgressiveStreamContents:
    data = bytes(data)
    if len(data) < PROMPT_LENGTH.size:
        raise ValueError("truncated progressive stream prompt length")
    descriptor = PROMPT_LENGTH.unpack_from(data, 0)[0]
    prompt_mode = PROMPT_MODE_RAW_UTF8 if descriptor & PROMPT_MODE_MASK else PROMPT_MODE_ZLIB
    prompt_len = descriptor & PROMPT_LENGTH_MASK
    prompt_start = PROMPT_LENGTH.size
    prompt_end = prompt_start + int(prompt_len)
    if prompt_end > len(data):
        raise ValueError("truncated progressive stream prompt payload")
    prompt_payload = data[prompt_start:prompt_end]
    try:
        if prompt_mode == PROMPT_MODE_RAW_UTF8:
            prompt = prompt_payload.decode("utf-8")
        else:
            prompt = zlib.decompress(prompt_payload).decode("utf-8")
    except (zlib.error, UnicodeDecodeError) as exc:
        raise ValueError("invalid progressive stream prompt payload") from exc
    return ProgressiveStreamContents(
        prompt=prompt,
        payload=data[prompt_end:],
        prompt_bytes=PROMPT_LENGTH.size + int(prompt_len),
        file_bytes=len(data),
        prompt_mode=prompt_mode_name(prompt_mode),
    )


def unpack_progressive_target_stream(data: bytes) -> ProgressiveStreamContents:
    data = bytes(data)
    if len(data) < TARGET_PAYLOAD_BITS.size:
        raise ValueError("truncated progressive target-payload header")
    target_payload_bits = TARGET_PAYLOAD_BITS.unpack_from(data, 0)[0]
    contents = unpack_progressive_stream(data[TARGET_PAYLOAD_BITS.size :])
    return ProgressiveStreamContents(
        prompt=contents.prompt,
        payload=contents.payload,
        prompt_bytes=contents.prompt_bytes,
        file_bytes=len(data),
        prompt_mode=contents.prompt_mode,
        target_payload_bits=int(target_payload_bits),
        target_header_bytes=TARGET_PAYLOAD_BITS.size,
    )


def write_progressive_stream(path: str, *, prompt: str, payload: bytes, fsync: bool = False) -> int:
    stream = pack_progressive_stream(prompt=prompt, payload=payload)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(stream)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    return len(stream)


def read_progressive_stream(path: str) -> ProgressiveStreamContents:
    with open(path, "rb") as handle:
        return unpack_progressive_stream(handle.read())


def write_progressive_target_stream(
    path: str,
    *,
    prompt: str,
    payload: bytes,
    target_payload_bits: int,
    fsync: bool = False,
) -> int:
    stream = pack_progressive_target_stream(
        prompt=prompt,
        payload=payload,
        target_payload_bits=int(target_payload_bits),
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(stream)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    return len(stream)


def read_progressive_target_stream(path: str) -> ProgressiveStreamContents:
    with open(path, "rb") as handle:
        return unpack_progressive_target_stream(handle.read())
