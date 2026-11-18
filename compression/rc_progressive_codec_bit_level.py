"""Bit-level progressive rate-control codec with two-pass CFG decode.

Encoding always uses the CFG-free entropy path so encoder and decoder can
reproduce the same probability model and bit selection. Decoding first
range-decodes transmitted bits with the same CFG-free path, then runs a second
generation pass with the requested CFG/tau and forces decoded bits back into
their selected positions.
"""

from __future__ import annotations

import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from codec.arithmetic_accel import RangeDecoder
from compression.hprc_core import (
    ProgressiveDecodeResult,
    ProgressiveEncodeResult,
    _append_prefix_seed_record,
    _autocast_for_model,
    _cfg_insertion_layer,
    _completion_stats,
    _copy_selected_bits,
    _decode_selected_into_sample,
    _empty_completion_stats,
    _kept_bit_count,
    _normalize_params,
    resolve_target_rate_params,
    _normalize_text_condition,
    _select_progressive,
    _sync_model,
    _target_payload_bits,
    _token_coverage_stats,
    derive_decode_seed,
    encode_progressive_rate_control,
)
from compression.rc_progressive_selector_bit_level import (
    BitScaleSelectionData,
    build_bit_scale_selection_data,
)
from infinity.models.infinity_ar_session import InfinityARSession


@torch.no_grad()
def decode_progressive_rate_control(
    *,
    model,
    vae,
    scale_schedule: Sequence[Tuple[int, int, int]],
    text_condition,
    prompt: str,
    payload: bytes,
    target_bpp: float,
    prompt_bits: int,
    ms: str,
    bit_depth: int,
    fixed_hw: Tuple[int, int],
    params: Optional[Mapping] = None,
    gt_ms_idx_Bl: Optional[Sequence[torch.Tensor]] = None,
) -> ProgressiveDecodeResult:
    params = resolve_target_rate_params(params, target_bpp)
    text_condition = _normalize_text_condition(text_condition)
    target_total_bits, target_payload_bits = _target_payload_bits(
        target_bpp,
        height=int(fixed_hw[0]),
        width=int(fixed_hw[1]),
        prompt_bits=int(prompt_bits),
    )
    remaining = max(0.0, float(target_payload_bits) - float(params["rate_overhead_bits"]))
    tolerance = float(params["budget_tolerance_bits"])
    seed_policy = str(params["seed_policy"])
    decode_seed = None
    if seed_policy != "prefix_hash":
        decode_seed = derive_decode_seed(
            seed_policy=seed_policy,
            prompt=prompt,
            target_payload_bits=target_payload_bits,
            ms=ms,
            bit_depth=bit_depth,
            scale_schedule=scale_schedule,
            params=params,
        )

    _sync_model(model)
    start = time.perf_counter()
    decoder = RangeDecoder(bytes(payload))
    per_scale_keep = [0] * len(scale_schedule)
    boundary_scale = -1
    full_scales = 0
    partial_bits = 0
    partial_tokens = 0
    full_tokens_in_partial_scale = 0
    empty_tokens_in_partial_scale = 0
    completion_stats = _empty_completion_stats()
    predicted_symbol_bits = 0.0
    prefix_seed_records: List[Tuple[int, int, int, bytes, int, bytes]] = []
    forced_bits_by_scale: Dict[int, torch.Tensor] = {}
    forced_data_by_scale: Dict[int, BitScaleSelectionData] = {}
    forced_kept_by_scale: Dict[int, Optional[torch.Tensor]] = {}

    # Phase 1: reproduce the encoder's entropy model exactly. This pass is
    # always CFG-free; it only reconstructs transmitted bits and their indices.
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
        ) as entropy_session:
            for si, _ in enumerate(scale_schedule):
                output = entropy_session.predict_next_scale()
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
                decoded_bits = torch.zeros(
                    (1, 1, int(data.height), int(data.width), int(data.bit_depth)),
                    device=output.logits.device,
                    dtype=torch.uint8,
                )
                _decode_selected_into_sample(decoder, decoded_bits, data, kept)
                _append_prefix_seed_record(
                    prefix_seed_records,
                    bits=decoded_bits,
                    data=data,
                    kept_bits=kept,
                )

                kept_count = _kept_bit_count(data, kept)
                forced_bits_by_scale[si] = decoded_bits
                forced_data_by_scale[si] = data
                forced_kept_by_scale[si] = kept
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
                entropy_session.commit_scale(decoded_bits)
            entropy_session.release_transformer_state()

    if seed_policy == "prefix_hash" and decode_seed is None:
        decode_seed = derive_decode_seed(
            seed_policy=seed_policy,
            prompt=prompt,
            target_payload_bits=target_payload_bits,
            ms=ms,
            bit_depth=bit_depth,
            scale_schedule=scale_schedule,
            params=params,
            prefix_records=prefix_seed_records,
        )

    # Phase 2: reconstruct with the requested CFG/tau. Known bits from phase 1
    # are forced into sampled scales. In prefix-hash mode, fully transmitted
    # prefix scales do not consume RNG; sampling starts at the first mixed scale.
    initial_seed = None if seed_policy == "prefix_hash" else decode_seed
    prefix_hash_seed_set = False
    with _autocast_for_model(model):
        with InfinityARSession(
            model,
            vae,
            list(scale_schedule),
            text_condition,
            seed=initial_seed,
            do_sample=bool(params["decode_do_sample"]),
            empty_cache_before_vae=bool(params["empty_cache_before_vae"]),
            decode_cfg=float(params["decode_cfg"]),
            decode_tau=float(params["decode_tau"]),
            cfg_insertion_layer=_cfg_insertion_layer(params),
        ) as generation_session:
            for si, _ in enumerate(scale_schedule):
                output = generation_session.predict_next_scale()
                if si in forced_bits_by_scale:
                    data = forced_data_by_scale[si]
                    kept = forced_kept_by_scale[si]
                    forced_bits = forced_bits_by_scale[si]
                    is_prefix_hash_full_scale = (
                        seed_policy == "prefix_hash"
                        and (boundary_scale < 0 or si < boundary_scale)
                    )
                    if is_prefix_hash_full_scale:
                        sampled_bits = forced_bits
                    else:
                        if seed_policy == "prefix_hash" and not prefix_hash_seed_set:
                            generation_session.set_seed(decode_seed)
                            prefix_hash_seed_set = True
                        sampled_bits = generation_session.sample_current_scale(output)
                        _copy_selected_bits(
                            source_bits=forced_bits,
                            target_bits=sampled_bits,
                            data=data,
                            kept_bits=kept,
                        )
                    if si == boundary_scale:
                        gt_scale = None if gt_ms_idx_Bl is None else gt_ms_idx_Bl[si]
                        completion_stats = _completion_stats(
                            vae=vae,
                            gt_scale=gt_scale,
                            mixed_bits=sampled_bits,
                            data=data,
                            kept_bits=kept,
                        )
                else:
                    sampled_bits = generation_session.sample_current_scale(output)
                generation_session.commit_scale(sampled_bits)
            image = generation_session.finish_image()

    # InfinityARSession returns the same channel order used by the ARPC session;
    # flip once more to match the RGB images used by raw eval metrics.
    if image.ndim == 4 and int(image.shape[-1]) == 3:
        image = image.flip(dims=(3,))

    _sync_model(model)
    decode_core_s = time.perf_counter() - start
    return ProgressiveDecodeResult(
        image_bhwc_u8=image,
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
        missing_bit_count=int(completion_stats["missing_bit_count"]),
        missing_bit_acc=float(completion_stats["missing_bit_acc"]),
        mixed_token_exact_rate=float(completion_stats["mixed_token_exact_rate"]),
        partial_token_exact_rate=float(completion_stats["partial_token_exact_rate"]),
        partial_token_hamming_mean=float(completion_stats["partial_token_hamming_mean"]),
        partial_token_hamming_hist=list(completion_stats["partial_token_hamming_hist"]),
        code_mse_mixed_gt=float(completion_stats["code_mse_mixed_gt"]),
        code_cosine_mixed_gt=float(completion_stats["code_cosine_mixed_gt"]),
        partial_code_mse_mixed_gt=float(completion_stats["partial_code_mse_mixed_gt"]),
        partial_code_cosine_mixed_gt=float(completion_stats["partial_code_cosine_mixed_gt"]),
        timings={"decode_core_s": float(decode_core_s)},
    )


__all__ = [
    "ProgressiveDecodeResult",
    "ProgressiveEncodeResult",
    "decode_progressive_rate_control",
    "encode_progressive_rate_control",
]
