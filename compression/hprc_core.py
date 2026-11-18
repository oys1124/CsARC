"""HPRC encoding, bit-budget allocation and deterministic seeds."""

from __future__ import annotations

import time
import hashlib
import struct
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from codec.arithmetic_accel import RangeDecoder, RangeEncoder
from compression.rc_progressive_profile import derive_rate_control_seed
from compression.rc_progressive_selector_bit_level import (
    BitScaleSelectionData,
    build_bit_scale_selection_data,
)
from infinity.models.infinity_ar_session import InfinityARSession


_PREFIX_HASH_PERSON = b"InfPrefixSeedV1"
_LENGTH_STRUCT = struct.Struct("<Q")
_SEED_SCALE_HEADER = struct.Struct("<IIQQ")
_MAX_SEED = (1 << 63) - 1


@dataclass
class ProgressiveEncodeResult:
    payload: bytes
    target_bpp: float
    target_total_bits: int
    target_payload_bits: int
    prompt_bits: int
    predicted_symbol_bits: float
    actual_payload_bits: int
    boundary_scale: int
    full_scales: int
    partial_positions: int
    per_scale_kept_pos: List[int]
    decode_seed: Optional[int]
    partial_bits: int = 0
    per_scale_kept_bits: List[int] = field(default_factory=list)
    partial_tokens: int = 0
    full_tokens_in_partial_scale: int = 0
    empty_tokens_in_partial_scale: int = 0
    missing_bit_count: int = 0
    missing_bit_acc: float = float("nan")
    mixed_token_exact_rate: float = float("nan")
    partial_token_exact_rate: float = float("nan")
    partial_token_hamming_mean: float = float("nan")
    partial_token_hamming_hist: List[int] = field(default_factory=list)
    code_mse_mixed_gt: float = float("nan")
    code_cosine_mixed_gt: float = float("nan")
    partial_code_mse_mixed_gt: float = float("nan")
    partial_code_cosine_mixed_gt: float = float("nan")
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProgressiveDecodeResult:
    image_bhwc_u8: torch.Tensor
    target_bpp: float
    target_total_bits: int
    target_payload_bits: int
    prompt_bits: int
    predicted_symbol_bits: float
    actual_payload_bits: int
    boundary_scale: int
    full_scales: int
    partial_positions: int
    per_scale_kept_pos: List[int]
    decode_seed: Optional[int]
    partial_bits: int = 0
    per_scale_kept_bits: List[int] = field(default_factory=list)
    partial_tokens: int = 0
    full_tokens_in_partial_scale: int = 0
    empty_tokens_in_partial_scale: int = 0
    missing_bit_count: int = 0
    missing_bit_acc: float = float("nan")
    mixed_token_exact_rate: float = float("nan")
    partial_token_exact_rate: float = float("nan")
    partial_token_hamming_mean: float = float("nan")
    partial_token_hamming_hist: List[int] = field(default_factory=list)
    code_mse_mixed_gt: float = float("nan")
    code_cosine_mixed_gt: float = float("nan")
    partial_code_mse_mixed_gt: float = float("nan")
    partial_code_cosine_mixed_gt: float = float("nan")
    timings: Dict[str, float] = field(default_factory=dict)


def _sync_model(model) -> None:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_for_model(model):
    model = getattr(model, "_orig_mod", model)
    device_type = None
    preferred_dtype = None
    for chunk in getattr(model, "block_chunks", []):
        for module in getattr(chunk, "module", []):
            attn = getattr(module, "sa", getattr(module, "attn", None))
            mat_qkv = getattr(attn, "mat_qkv", None)
            weight = getattr(mat_qkv, "weight", None)
            if weight is None:
                continue
            if weight.device.type == "cuda":
                device_type = "cuda"
            if weight.dtype in (torch.bfloat16, torch.float16):
                preferred_dtype = weight.dtype
                break
        if preferred_dtype is not None:
            break
    if preferred_dtype is None:
        for module in getattr(model, "unregistered_blocks", []):
            attn = getattr(module, "sa", getattr(module, "attn", None))
            mat_qkv = getattr(attn, "mat_qkv", None)
            weight = getattr(mat_qkv, "weight", None)
            if weight is None:
                continue
            if weight.device.type == "cuda":
                device_type = "cuda"
            if weight.dtype in (torch.bfloat16, torch.float16):
                preferred_dtype = weight.dtype
                break
    for name, param in model.named_parameters():
        if param.device.type == "cuda":
            device_type = "cuda"
        if "mat_qkv" in name and param.dtype in (torch.bfloat16, torch.float16):
            preferred_dtype = param.dtype
            break
    if preferred_dtype is None:
        for param in model.parameters():
            if param.device.type == "cuda":
                device_type = "cuda"
            if param.dtype in (torch.bfloat16, torch.float16):
                preferred_dtype = param.dtype
                break
    if device_type != "cuda" or preferred_dtype is None:
        return nullcontext()
    if preferred_dtype == torch.bfloat16:
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if preferred_dtype == torch.float16:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def _normalize_text_condition(text_condition):
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_condition
    if not hasattr(lens, "shape"):
        lens = torch.as_tensor(lens, dtype=torch.int32, device=cu_seqlens_k.device)
    return kv_compact, lens, cu_seqlens_k, max_seqlen_k


def _normalize_params(params: Optional[Mapping]) -> Dict:
    out = dict(params or {})
    out.setdefault("scale_weight", "area")
    out.setdefault("bit_select_order", "high")
    out.setdefault("prob_tau", 1.0)
    out.setdefault("budget_rate_scale", 1.0)
    out.setdefault("decode_cfg", 1.0)
    out.setdefault("decode_tau", 1.0)
    out.setdefault("cfg_insertion_layer", [0])
    out.setdefault("rate_overhead_bits", 8.0)
    out.setdefault("budget_tolerance_bits", 1e-6)
    out.setdefault("seed_policy", "rate_hash")
    out.setdefault("hash_seed_salt", "InfinityCodecRateControlSeedV1")
    out.setdefault("decode_do_sample", True)
    out.setdefault("empty_cache_before_vae", True)
    return out


def _parse_target_schedule(value, *, name: str, allow_zero: bool) -> List[Tuple[float, float]]:
    """Parse ``target_lower_bound:value`` pairs for target-dependent calibration."""
    if value is None or value == "":
        return []
    if isinstance(value, Mapping):
        items = [(float(key), float(item)) for key, item in value.items()]
    elif isinstance(value, str):
        items = []
        for raw_item in value.split(","):
            item = raw_item.strip()
            if not item:
                continue
            parts = item.split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"invalid {name} entry {item!r}; expected target_lower_bound:value"
                )
            items.append((float(parts[0].strip()), float(parts[1].strip())))
    else:
        items = [(float(key), float(item)) for key, item in value]
    items.sort(key=lambda pair: pair[0])
    if any(lower < 0.0 for lower, _ in items):
        raise ValueError(f"{name} target lower bounds must be non-negative")
    if any(value < 0.0 or (not allow_zero and value <= 0.0) for _, value in items):
        requirement = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} values must be {requirement}")
    if len({lower for lower, _ in items}) != len(items):
        raise ValueError(f"{name} contains duplicate target lower bounds")
    return items


def resolve_target_rate_params(params: Optional[Mapping], target_bpp: float) -> Dict:
    """Resolve optional piecewise rate-control parameters for one target BPP.

    Schedule entries use ``target_lower_bound:value`` and the last entry whose
    lower bound is no larger than ``target_bpp`` is selected.  A schedule can
    start at 0.0 to cover every target; otherwise the scalar default remains in
    effect below its first lower bound.
    """
    out = _normalize_params(params)
    target = float(target_bpp)
    for schedule_key, parameter_key, allow_zero in (
        ("budget_rate_scale_schedule", "budget_rate_scale", False),
        ("rate_overhead_schedule", "rate_overhead_bits", True),
    ):
        schedule = _parse_target_schedule(out.get(schedule_key), name=schedule_key, allow_zero=allow_zero)
        for lower_bound, value in schedule:
            if target >= lower_bound:
                out[parameter_key] = value
            else:
                break
    return out


def _cfg_insertion_layer(params: Mapping) -> List[int]:
    value = params.get("cfg_insertion_layer", [0])
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _target_payload_bits(target_bpp: float, *, height: int, width: int, prompt_bits: int) -> Tuple[int, int]:
    total = max(0, int(round(float(target_bpp) * float(height * width))))
    payload = max(0, total - int(prompt_bits))
    return total, payload


def _scale_bits_for_commit(gt_scale: torch.Tensor, data: BitScaleSelectionData) -> torch.Tensor:
    return gt_scale.reshape(1, int(data.height), int(data.width), int(data.bit_depth)).unsqueeze(1).to(
        dtype=torch.uint8
    )


def _encode_selected(
    encoder: RangeEncoder,
    gt_scale: torch.Tensor,
    data: BitScaleSelectionData,
    kept_bits: Optional[torch.Tensor],
) -> None:
    if kept_bits is not None and int(kept_bits.numel()) == 0:
        return
    gt_bits = _scale_bits_for_commit(gt_scale, data)
    flat = gt_bits[:, 0].reshape(1, data.num_bits)
    p1_flat = data.p1.reshape(1, data.num_bits)
    if kept_bits is None:
        bits_t = flat
        prob_t = p1_flat
    else:
        bits_t = flat.index_select(1, kept_bits)
        prob_t = p1_flat.index_select(1, kept_bits)
    bits = bits_t.reshape(-1).detach().cpu().numpy().astype(np.uint8)
    probabilities = prob_t.reshape(-1).detach().cpu().numpy().astype(np.float32)
    encoder.encode(bits, probabilities)


def _decode_selected_into_sample(
    decoder: RangeDecoder,
    sampled_bits: torch.Tensor,
    data: BitScaleSelectionData,
    kept_bits: Optional[torch.Tensor],
) -> None:
    if kept_bits is not None and int(kept_bits.numel()) == 0:
        return
    p1_flat = data.p1.reshape(1, data.num_bits)
    prob_t = p1_flat if kept_bits is None else p1_flat.index_select(1, kept_bits)
    probabilities = prob_t.reshape(-1).detach().cpu().numpy().astype(np.float32)
    decoded = decoder.decode(probabilities)
    decoded_len = data.num_bits if kept_bits is None else int(kept_bits.numel())
    decoded_t = torch.from_numpy(
        np.asarray(decoded, dtype=np.uint8).reshape(1, int(decoded_len))
    ).to(device=sampled_bits.device, dtype=torch.uint8)
    flat = sampled_bits[:, 0].reshape(1, data.num_bits)
    if kept_bits is None:
        flat.copy_(decoded_t)
    else:
        flat.index_copy_(1, kept_bits.to(sampled_bits.device), decoded_t)


def _copy_selected_bits(
    *,
    source_bits: torch.Tensor,
    target_bits: torch.Tensor,
    data: BitScaleSelectionData,
    kept_bits: Optional[torch.Tensor],
) -> None:
    if kept_bits is not None and int(kept_bits.numel()) == 0:
        return
    src_flat = source_bits[:, 0].reshape(1, data.num_bits)
    dst_flat = target_bits[:, 0].reshape(1, data.num_bits)
    if kept_bits is None:
        dst_flat.copy_(src_flat.to(target_bits.device))
    else:
        selected = src_flat.index_select(1, kept_bits.to(source_bits.device))
        dst_flat.index_copy_(1, kept_bits.to(target_bits.device), selected.to(target_bits.device))


def _append_prefix_seed_record(
    records: List[Tuple[int, int, int, bytes, int, bytes]],
    *,
    bits: torch.Tensor,
    data: BitScaleSelectionData,
    kept_bits: Optional[torch.Tensor],
) -> None:
    flat = bits[:, 0].reshape(1, data.num_bits)
    if kept_bits is None:
        positions_bytes = b"ALL_BITS"
        selected_t = flat
    else:
        kept_cpu = kept_bits.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
        positions_bytes = kept_cpu.tobytes()
        selected_t = flat.index_select(1, kept_bits.to(bits.device))
    selected = selected_t.reshape(-1).detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
    if selected.size and np.any(selected > 1):
        raise ValueError("prefix_hash seed requires binary token indices")
    packed_bits = np.packbits(selected, bitorder="big").tobytes()
    records.append(
        (
            int(data.scale_index),
            int(data.num_bits),
            int(data.bit_depth),
            positions_bytes,
            int(selected.size),
            packed_bits,
        )
    )


def _update_hash_bytes(hasher, payload: bytes) -> None:
    hasher.update(_LENGTH_STRUCT.pack(len(payload)))
    hasher.update(payload)


def _derive_progressive_prefix_hash_seed(
    *,
    prompt: str,
    scale_schedule,
    records: Sequence[Tuple[int, int, int, bytes, int, bytes]],
    salt: str,
) -> int:
    hasher = hashlib.blake2b(digest_size=16, person=_PREFIX_HASH_PERSON)
    _update_hash_bytes(hasher, str(salt).encode("utf-8"))
    _update_hash_bytes(hasher, str(prompt).encode("utf-8"))
    for scale in scale_schedule:
        hasher.update(bytes(int(v) & 0xFF for v in scale))
    for scale_index, num_positions, bit_depth, positions_bytes, bit_count, packed_bits in records:
        hasher.update(
            _SEED_SCALE_HEADER.pack(
                int(scale_index),
                int(bit_depth),
                int(num_positions),
                int(bit_count),
            )
        )
        _update_hash_bytes(hasher, positions_bytes)
        _update_hash_bytes(hasher, packed_bits)
    return int.from_bytes(hasher.digest()[:8], "little") & _MAX_SEED


def _select_progressive(
    data: BitScaleSelectionData,
    remaining_bits: float,
    *,
    tolerance_bits: float,
    bit_select_order: str,
) -> Tuple[Optional[torch.Tensor], float, bool]:
    if data.full_rate_bits <= float(remaining_bits) + float(tolerance_bits):
        return None, float(data.full_rate_bits), False
    order = str(bit_select_order).lower()
    if order not in {"high", "low", "sequential", "raster"}:
        raise ValueError(
            f"unsupported bit_select_order={bit_select_order!r}; expected high, low, sequential, or raster"
        )
    if order in {"sequential", "raster"}:
        fixed_order = torch.arange(data.num_bits, device=data.rate_per_bit.device, dtype=torch.long)
        kept = data.select_under_budget(
            remaining_bits,
            tolerance_bits=tolerance_bits,
            order=fixed_order,
        )
    else:
        kept = data.select_under_budget(
            remaining_bits,
            tolerance_bits=tolerance_bits,
            descending=(order == "high"),
        )
    selected_rate = (
        float(data.rate_per_bit.index_select(0, kept).sum().item())
        if int(kept.numel()) > 0
        else 0.0
    )
    return kept, selected_rate, True


def _kept_bit_count(data: BitScaleSelectionData, kept_bits: Optional[torch.Tensor]) -> int:
    return int(data.num_bits) if kept_bits is None else int(kept_bits.numel())


def _token_coverage_stats(
    data: BitScaleSelectionData,
    kept_bits: Optional[torch.Tensor],
) -> Tuple[int, int, int]:
    """Return partial/full/empty token counts under a bit-level keep set."""
    num_positions = int(data.num_positions)
    depth = int(data.bit_depth)
    if kept_bits is None:
        return 0, num_positions, 0
    if int(kept_bits.numel()) == 0:
        return 0, 0, num_positions
    pos_idx = torch.div(kept_bits, depth, rounding_mode="floor")
    counts = torch.bincount(pos_idx, minlength=num_positions)
    partial_tokens = int(((counts > 0) & (counts < depth)).sum().item())
    full_tokens = int((counts >= depth).sum().item())
    empty_tokens = int((counts == 0).sum().item())
    return partial_tokens, full_tokens, empty_tokens


def _flatten_lfq_codes(codes: torch.Tensor) -> torch.Tensor:
    codes = codes.float()
    if codes.ndim == 5:
        if int(codes.shape[2]) != 1:
            raise ValueError(f"expected single temporal axis in LFQ codes, got {tuple(codes.shape)}")
        codes = codes[:, :, 0]
    if codes.ndim == 4:
        return codes.permute(0, 2, 3, 1).reshape(-1, int(codes.shape[1]))
    return codes.reshape(-1, int(codes.shape[-1]))


def _empty_completion_stats() -> Dict:
    return {
        "missing_bit_count": 0,
        "missing_bit_acc": float("nan"),
        "mixed_token_exact_rate": float("nan"),
        "partial_token_exact_rate": float("nan"),
        "partial_token_hamming_mean": float("nan"),
        "partial_token_hamming_hist": [],
        "code_mse_mixed_gt": float("nan"),
        "code_cosine_mixed_gt": float("nan"),
        "partial_code_mse_mixed_gt": float("nan"),
        "partial_code_cosine_mixed_gt": float("nan"),
    }


def _completion_stats(
    *,
    vae,
    gt_scale: Optional[torch.Tensor],
    mixed_bits: torch.Tensor,
    data: BitScaleSelectionData,
    kept_bits: Optional[torch.Tensor],
) -> Dict:
    if gt_scale is None:
        return _empty_completion_stats()

    gt_bits = _scale_bits_for_commit(gt_scale, data).to(device=mixed_bits.device, dtype=torch.uint8)
    mixed_flat = mixed_bits[:, 0].reshape(1, int(data.num_positions), int(data.bit_depth))
    gt_flat = gt_bits[:, 0].reshape(1, int(data.num_positions), int(data.bit_depth))
    mixed_bits_1d = mixed_flat.reshape(-1)
    gt_bits_1d = gt_flat.reshape(-1)

    keep_mask = torch.zeros((int(data.num_bits),), device=mixed_bits.device, dtype=torch.bool)
    if kept_bits is None:
        keep_mask.fill_(True)
    elif int(kept_bits.numel()) > 0:
        keep_mask.index_fill_(0, kept_bits.to(mixed_bits.device), True)
    missing_mask = ~keep_mask
    missing_bit_count = int(missing_mask.sum().item())
    if missing_bit_count > 0:
        missing_bit_acc = float((mixed_bits_1d[missing_mask] == gt_bits_1d[missing_mask]).float().mean().item())
    else:
        missing_bit_acc = float("nan")

    token_equal = (mixed_flat == gt_flat).all(dim=2).reshape(-1)
    hamming = (mixed_flat != gt_flat).sum(dim=2).reshape(-1)
    mixed_token_exact_rate = float(token_equal.float().mean().item())

    depth = int(data.bit_depth)
    if kept_bits is None:
        counts = torch.full((int(data.num_positions),), depth, device=mixed_bits.device, dtype=torch.long)
    elif int(kept_bits.numel()) == 0:
        counts = torch.zeros((int(data.num_positions),), device=mixed_bits.device, dtype=torch.long)
    else:
        pos_idx = torch.div(kept_bits.to(mixed_bits.device), depth, rounding_mode="floor")
        counts = torch.bincount(pos_idx, minlength=int(data.num_positions))
    partial_mask = (counts > 0) & (counts < depth)
    if bool(partial_mask.any().item()):
        partial_hamming = hamming[partial_mask]
        partial_token_exact_rate = float(token_equal[partial_mask].float().mean().item())
        partial_token_hamming_mean = float(partial_hamming.float().mean().item())
        partial_token_hamming_hist = [
            int(value) for value in torch.bincount(partial_hamming, minlength=depth + 1).detach().cpu().tolist()
        ]
    else:
        partial_token_exact_rate = float("nan")
        partial_token_hamming_mean = float("nan")
        partial_token_hamming_hist = [0] * (depth + 1)

    stats = {
        "missing_bit_count": missing_bit_count,
        "missing_bit_acc": missing_bit_acc,
        "mixed_token_exact_rate": mixed_token_exact_rate,
        "partial_token_exact_rate": partial_token_exact_rate,
        "partial_token_hamming_mean": partial_token_hamming_mean,
        "partial_token_hamming_hist": partial_token_hamming_hist,
        "code_mse_mixed_gt": float("nan"),
        "code_cosine_mixed_gt": float("nan"),
        "partial_code_mse_mixed_gt": float("nan"),
        "partial_code_cosine_mixed_gt": float("nan"),
    }

    if vae is None:
        return stats
    mixed_codes = _flatten_lfq_codes(vae.quantizer.lfq.indices_to_codes(mixed_bits, label_type="bit_label"))
    gt_codes = _flatten_lfq_codes(vae.quantizer.lfq.indices_to_codes(gt_bits, label_type="bit_label"))
    stats["code_mse_mixed_gt"] = float((mixed_codes - gt_codes).pow(2).mean().item())
    stats["code_cosine_mixed_gt"] = float(F.cosine_similarity(mixed_codes, gt_codes, dim=1).mean().item())
    if bool(partial_mask.any().item()):
        partial_codes = mixed_codes[partial_mask]
        partial_gt_codes = gt_codes[partial_mask]
        stats["partial_code_mse_mixed_gt"] = float((partial_codes - partial_gt_codes).pow(2).mean().item())
        stats["partial_code_cosine_mixed_gt"] = float(
            F.cosine_similarity(partial_codes, partial_gt_codes, dim=1).mean().item()
        )
    return stats


def derive_decode_seed(
    *,
    seed_policy: str,
    prompt: str,
    target_payload_bits: int,
    ms: str,
    bit_depth: int,
    scale_schedule,
    params: Mapping,
    prefix_records: Optional[Sequence[Tuple[int, int, int, bytes, int, bytes]]] = None,
) -> Optional[int]:
    seed_policy = str(seed_policy)
    if seed_policy == "none":
        return None
    if seed_policy == "fixed":
        return int(params.get("fixed_seed", 0))
    if seed_policy == "prefix_hash":
        return _derive_progressive_prefix_hash_seed(
            prompt=prompt,
            scale_schedule=scale_schedule,
            records=prefix_records or [],
            salt=str(params.get("hash_seed_salt", "InfinityCodecPrefixSeedV1")),
        )
    if seed_policy == "rate_hash":
        return derive_rate_control_seed(
            prompt=prompt,
            target_payload_bits=int(target_payload_bits),
            ms=str(ms),
            bit_depth=int(bit_depth),
            scale_schedule=scale_schedule,
            salt=str(params.get("hash_seed_salt", "InfinityCodecRateControlSeedV1")),
        )
    raise ValueError(f"unsupported seed_policy={seed_policy!r}")


@torch.no_grad()
def encode_progressive_rate_control(
    *,
    model,
    vae,
    scale_schedule: Sequence[Tuple[int, int, int]],
    text_condition,
    prompt: str,
    gt_ms_idx_Bl: Sequence[torch.Tensor],
    target_bpp: float,
    prompt_bits: int,
    ms: str,
    bit_depth: int,
    fixed_hw: Tuple[int, int],
    params: Optional[Mapping] = None,
) -> ProgressiveEncodeResult:
    params = resolve_target_rate_params(params, target_bpp)
    text_condition = _normalize_text_condition(text_condition)
    target_total_bits, target_payload_bits = _target_payload_bits(
        target_bpp, height=int(fixed_hw[0]), width=int(fixed_hw[1]), prompt_bits=int(prompt_bits)
    )
    remaining = max(0.0, float(target_payload_bits) - float(params["rate_overhead_bits"]))
    tolerance = float(params["budget_tolerance_bits"])

    _sync_model(model)
    start = time.perf_counter()
    encoder = RangeEncoder()
    per_scale_keep = [0] * len(scale_schedule)
    boundary_scale = -1
    full_scales = 0
    partial_bits = 0
    partial_tokens = 0
    full_tokens_in_partial_scale = 0
    empty_tokens_in_partial_scale = 0
    predicted_symbol_bits = 0.0
    prefix_seed_records: List[Tuple[int, int, int, bytes, int, bytes]] = []
    with _autocast_for_model(model):
        with InfinityARSession(
            model,
            vae,
            list(scale_schedule),
            text_condition,
            seed=None,
            do_sample=False,
            empty_cache_before_vae=False,
            decode_cfg=1.0,
            decode_tau=1.0,
            cfg_insertion_layer=_cfg_insertion_layer(params),
        ) as session:
            for si, _ in enumerate(scale_schedule):
                output = session.predict_next_scale()
                data = build_bit_scale_selection_data(
                    logits=output.logits,
                    scale_index=si,
                    patch_shape=output.patch_shape,
                    bit_depth=int(bit_depth),
                    fixed_hw=fixed_hw,
                    ms=ms,
                    params=params,
                )
                kept, selected_rate, boundary = _select_progressive(
                    data,
                    remaining,
                    tolerance_bits=tolerance,
                    bit_select_order=str(params["bit_select_order"]),
                )
                _encode_selected(encoder, gt_ms_idx_Bl[si], data, kept)
                _append_prefix_seed_record(
                    prefix_seed_records,
                    bits=_scale_bits_for_commit(gt_ms_idx_Bl[si], data).to(output.logits.device),
                    data=data,
                    kept_bits=kept,
                )
                kept_count = _kept_bit_count(data, kept)
                per_scale_keep[si] = int(kept_count)
                predicted_symbol_bits += float(selected_rate)
                remaining = max(0.0, remaining - float(selected_rate))
                if boundary:
                    boundary_scale = int(si)
                    partial_bits = int(kept_count)
                    (
                        partial_tokens,
                        full_tokens_in_partial_scale,
                        empty_tokens_in_partial_scale,
                    ) = _token_coverage_stats(data, kept)
                    break
                full_scales += 1
                session.commit_scale(_scale_bits_for_commit(gt_ms_idx_Bl[si], data))

    payload = encoder.finish()
    _sync_model(model)
    encode_core_s = time.perf_counter() - start
    decode_seed = derive_decode_seed(
        seed_policy=str(params["seed_policy"]),
        prompt=prompt,
        target_payload_bits=target_payload_bits,
        ms=ms,
        bit_depth=bit_depth,
        scale_schedule=scale_schedule,
        params=params,
        prefix_records=prefix_seed_records,
    )
    return ProgressiveEncodeResult(
        payload=payload,
        target_bpp=float(target_bpp),
        target_total_bits=int(target_total_bits),
        target_payload_bits=int(target_payload_bits),
        prompt_bits=int(prompt_bits),
        predicted_symbol_bits=float(predicted_symbol_bits),
        actual_payload_bits=int(len(payload) * 8),
        boundary_scale=int(boundary_scale),
        full_scales=int(full_scales),
        partial_positions=int(partial_bits),
        per_scale_kept_pos=per_scale_keep,
        decode_seed=decode_seed,
        partial_bits=int(partial_bits),
        per_scale_kept_bits=per_scale_keep,
        partial_tokens=int(partial_tokens),
        full_tokens_in_partial_scale=int(full_tokens_in_partial_scale),
        empty_tokens_in_partial_scale=int(empty_tokens_in_partial_scale),
        timings={"encode_core_s": float(encode_core_s)},
    )
