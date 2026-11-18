"""Fixed-profile zero-header progressive stream.

The codec profile (1024x1024, bit depth, scale schedule, tau/CFG, model version and
RNG policy) is shared by encoder and decoder. Every stream begins with its
caption; each scale adds a two-byte descriptor and one byte-aligned payload.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Fixed 1024-square codec profile.
FIXED_STREAM_DECODE_SEED = 0
MODE_RAW = 0
MODE_ARITHMETIC = 1
PROFILE_INFINITY_1024_D32 = 1
PROFILE_INFINITY_1024_D16 = 2
PROFILE_INFINITY_1024_SCHEDULE = [(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 6, 6), (1, 8, 8),
                                  (1, 12, 12), (1, 16, 16), (1, 20, 20), (1, 24, 24), (1, 32, 32),
                                  (1, 40, 40), (1, 48, 48), (1, 64, 64)]
PROFILE_SCHEDULES = {
    PROFILE_INFINITY_1024_D32: PROFILE_INFINITY_1024_SCHEDULE,
    PROFILE_INFINITY_1024_D16: PROFILE_INFINITY_1024_SCHEDULE
}


def _normalize_schedule(schedule) -> List[Tuple[int, int, int]]:
    return [tuple(int(value) for value in scale) for scale in schedule]


def resolve_profile(width: int, height: int, bit_depth: int, scale_schedule) -> int:
    schedule = _normalize_schedule(scale_schedule)
    if (int(width) == 1024 and int(height) == 1024 and schedule == PROFILE_INFINITY_1024_SCHEDULE):
        if int(bit_depth) == 32:
            return PROFILE_INFINITY_1024_D32
        if int(bit_depth) == 16:
            return PROFILE_INFINITY_1024_D16
    raise ValueError("RangeCoder stream supports only the 1024x1024, d16/d32, 13-scale profile")


PROMPT_LENGTH = struct.Struct("<H")
SCALE_DESCRIPTOR = struct.Struct("<H")
MODE_MASK = 1 << 15
LENGTH_MASK = MODE_MASK - 1
PROMPT_MODE_ZLIB = 0
PROMPT_MODE_RAW_UTF8 = 1


@dataclass
class ZeroHeaderRawStreamContents:
    metadata: Dict[str, Any]
    prompt: str
    scale_payloads: List[bytes]
    scale_modes: List[int]
    prefix_bytes: List[int]
    header_bytes: int
    file_bytes: int


def select_zero_header_prompt_payload(prompt: str) -> Tuple[int, bytes]:
    """Return the shorter zlib or UTF-8 caption payload.

    The caption mode is encoded in the high bit of its length descriptor.
    """
    raw_payload = str(prompt).encode("utf-8")
    zlib_payload = zlib.compress(raw_payload)

    candidates = []
    if len(zlib_payload) <= LENGTH_MASK:
        candidates.append((PROMPT_MODE_ZLIB, zlib_payload))
    if len(raw_payload) <= LENGTH_MASK:
        candidates.append((PROMPT_MODE_RAW_UTF8, raw_payload))
    if not candidates:
        raise ValueError("prompt payload exceeds zero-header uint15 length")

    return min(candidates, key=lambda item: len(item[1]))


def decode_zero_header_prompt_payload(mode: int, payload: bytes) -> str:
    mode = int(mode)
    if mode == PROMPT_MODE_ZLIB:
        return zlib.decompress(payload).decode("utf-8")
    if mode == PROMPT_MODE_RAW_UTF8:
        return payload.decode("utf-8")
    raise ValueError(f"unsupported zero-header prompt mode={mode}")


def build_zero_header_stream(
    *,
    width: int,
    height: int,
    bit_depth: int,
    scale_schedule,
    prompt: str,
    scale_payloads: Sequence[bytes],
    scale_modes: Sequence[int],
) -> Tuple[bytes, List[int]]:
    profile_id = resolve_profile(width, height, bit_depth, scale_schedule)
    schedule = PROFILE_SCHEDULES[profile_id]
    if len(scale_payloads) != len(scale_modes) or len(scale_payloads) > len(schedule):
        raise ValueError("invalid zero-header scale payload counts")
    output = bytearray()
    prompt_mode, prompt_payload = select_zero_header_prompt_payload(prompt)
    descriptor = len(prompt_payload)
    if prompt_mode == PROMPT_MODE_RAW_UTF8:
        descriptor |= MODE_MASK
    output.extend(PROMPT_LENGTH.pack(descriptor))
    output.extend(prompt_payload)

    prefix_bytes: List[int] = []
    for scale_index, (payload, mode) in enumerate(zip(scale_payloads, scale_modes)):
        payload = bytes(payload)
        if len(payload) > LENGTH_MASK:
            raise ValueError("scale payload exceeds 15-bit byte length")
        if int(mode) not in (MODE_RAW, MODE_ARITHMETIC):
            raise ValueError(f"invalid zero-header scale mode={mode}")
        symbol_count = (schedule[scale_index][0] * schedule[scale_index][1] *
                        schedule[scale_index][2] * int(bit_depth))
        if int(mode) == MODE_RAW and len(payload) * 8 != symbol_count:
            raise ValueError("raw zero-header payload does not match profile symbol count")
        descriptor = len(payload) | (MODE_MASK if int(mode) == MODE_ARITHMETIC else 0)
        output.extend(SCALE_DESCRIPTOR.pack(descriptor))
        output.extend(payload)
        prefix_bytes.append(len(output))
    return bytes(output), prefix_bytes


def parse_zero_header_stream(
    data: bytes,
    *,
    max_scale: Optional[int] = None,
    bit_depth: int = 32,
) -> ZeroHeaderRawStreamContents:
    bit_depth = int(bit_depth)
    profile_id = resolve_profile(1024, 1024, bit_depth, PROFILE_INFINITY_1024_SCHEDULE)
    if len(data) < PROMPT_LENGTH.size:
        raise ValueError("truncated standalone prompt length")
    prompt_descriptor = PROMPT_LENGTH.unpack_from(data, 0)[0]
    prompt_mode = PROMPT_MODE_RAW_UTF8 if prompt_descriptor & MODE_MASK else PROMPT_MODE_ZLIB
    prompt_len = prompt_descriptor & LENGTH_MASK
    prompt_end = PROMPT_LENGTH.size + int(prompt_len)
    if prompt_end > len(data):
        raise ValueError("truncated standalone prompt payload")
    try:
        prompt = decode_zero_header_prompt_payload(
            prompt_mode,
            data[PROMPT_LENGTH.size:prompt_end],
        )
    except (zlib.error, UnicodeDecodeError) as exc:
        raise ValueError("invalid standalone prompt payload") from exc
    offset = prompt_end
    header_bytes = offset

    schedule = PROFILE_SCHEDULES[profile_id]
    scale_payloads: List[bytes] = []
    scale_modes: List[int] = []
    prefix_bytes: List[int] = []
    while offset < len(data):
        if len(scale_payloads) >= len(schedule):
            raise ValueError("zero-header stream contains too many scale sections")
        descriptor_end = offset + SCALE_DESCRIPTOR.size
        if descriptor_end > len(data):
            raise ValueError("truncated zero-header scale descriptor")
        descriptor = SCALE_DESCRIPTOR.unpack_from(data, offset)[0]
        mode = MODE_ARITHMETIC if descriptor & MODE_MASK else MODE_RAW
        payload_len = descriptor & LENGTH_MASK
        payload_end = descriptor_end + int(payload_len)
        if payload_end > len(data):
            raise ValueError("truncated zero-header scale payload")
        payload = bytes(data[descriptor_end:payload_end])
        symbol_count = (schedule[len(scale_payloads)][0] * schedule[len(scale_payloads)][1] *
                        schedule[len(scale_payloads)][2] * bit_depth)
        if mode == MODE_RAW and len(payload) * 8 != symbol_count:
            raise ValueError("raw zero-header scale length does not match fixed profile")
        scale_payloads.append(payload)
        scale_modes.append(mode)
        prefix_bytes.append(payload_end)
        offset = payload_end
        if max_scale is not None and len(scale_payloads) - 1 >= int(max_scale):
            break

    metadata = {
        "format_version": 3,
        "profile_id": profile_id,
        "width": 1024,
        "height": 1024,
        "bit_depth": bit_depth,
        "scale_schedule": [list(scale) for scale in schedule],
        "prob_tau": 0.5,
        "decode_cfg": 3.0,
        "decode_tau": 0.5,
        "cfg_insertion_layer": [0],
        "rng_mode": "external_profile",
        "decode_seed": None,
        "prompt_transport": "standalone_zlib",
    }
    return ZeroHeaderRawStreamContents(
        metadata=metadata,
        prompt=prompt,
        scale_payloads=scale_payloads,
        scale_modes=scale_modes,
        prefix_bytes=prefix_bytes,
        header_bytes=header_bytes,
        file_bytes=len(data),
    )


def write_zero_header_stream(path: str, payload: bytes, fsync: bool = False) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def read_zero_header_stream(
    path: str,
    *,
    max_scale: Optional[int] = None,
    bit_depth: int = 32,
):
    with open(path, "rb") as handle:
        return parse_zero_header_stream(
            handle.read(),
            max_scale=max_scale,
            bit_depth=bit_depth,
        )
