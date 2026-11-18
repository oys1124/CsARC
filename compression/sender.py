"""Scale-wise CSA RangeCoder encoding."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from codec.arithmetic_accel import RangeEncoder
from compression.util import CrossAttnBlock, encode_prompt
from compression.raw_stream_io_v3 import (
    MODE_ARITHMETIC,
    MODE_RAW,
    build_zero_header_stream,
    write_zero_header_stream,
)

__all__ = ['StreamEncodeResult', 'encoding_to_stream', 'get_prob']


@dataclass
class StreamEncodeResult:
    stream_path: str
    stream_prefix_bytes: List[int]
    stream_file_bytes: int
    coding_prefix_bits: List[int]
    scale_modes: List[int]
    timings: Dict[str, float]


def _sync_model(model) -> None:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_prob(
    infinity,
    vae,
    vae_scale_schedule,
    prompt,
    text_tokenizer,
    text_encoder,
    gt_ls_Bl,
    tau_list=0.5,
    cfg_insertion_layer=[0],
    max_scale=None,
):
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(vae_scale_schedule)
    if max_scale is None:
        final_probability_scale = len(vae_scale_schedule) - 1
    else:
        final_probability_scale = int(max_scale)
        if final_probability_scale < 0 or final_probability_scale >= len(vae_scale_schedule):
            raise ValueError(f"max_scale={max_scale} is outside [0, {len(vae_scale_schedule) - 1}]")
    label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, prompt)
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
    bs = B = 1
    kv_compact = infinity.text_norm(kv_compact)
    sos = cond_BD = infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
    kv_compact = infinity.text_proj_for_ca(kv_compact)
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + infinity.pos_start.expand(bs, 1, -1)
    with torch.amp.autocast('cuda', enabled=False):
        cond_BD_or_gss = infinity.shared_ada_lin(cond_BD.float()).float().contiguous()
    for b in infinity.unregistered_blocks:
        (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
    abs_cfg_insertion_layers = []
    add_cfg_on_logits, add_cfg_on_probs = False, False
    leng = len(infinity.unregistered_blocks)
    for item in cfg_insertion_layer:
        if item == 0:  # add cfg on logits
            add_cfg_on_logits = True
        elif item == 1:  # add cfg on probs
            add_cfg_on_probs = True
        elif item < 0:  # determine to add cfg at item-th layer's output
            assert leng + item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={infinity.num_block_chunks}'
            abs_cfg_insertion_layers.append(leng + item)
        else:
            raise ValueError(f'cfg_insertion_layer: {item} is not valid')

    summed_codes = 0
    cur_L = 0
    prob_list = []
    for si, pn in enumerate(vae_scale_schedule[:final_probability_scale + 1]):
        cur_L += np.array(pn).prod()
        need_to_pad = 0
        attn_fn = None
        if infinity.use_flex_attn:
            attn_fn = infinity.attn_fn_compile_dict.get(tuple(vae_scale_schedule[:(si + 1)]), None)
        layer_idx = 0
        for block_idx, b in enumerate(infinity.block_chunks):
            if infinity.add_lvl_embeding_only_first_block and block_idx == 0:
                last_stage = infinity.add_lvl_embeding(last_stage,
                                                       si,
                                                       vae_scale_schedule,
                                                       need_to_pad=need_to_pad)
            if not infinity.add_lvl_embeding_only_first_block:
                last_stage = infinity.add_lvl_embeding(last_stage,
                                                       si,
                                                       vae_scale_schedule,
                                                       need_to_pad=need_to_pad)

            for m in b.module:
                last_stage = m(x=last_stage,
                               cond_BD=cond_BD_or_gss,
                               ca_kv=ca_kv,
                               attn_bias_or_two_vector=None,
                               attn_fn=attn_fn,
                               scale_schedule=vae_scale_schedule,
                               rope2d_freqs_grid=infinity.rope2d_freqs_grid,
                               scale_ind=si)
                layer_idx += 1
        logits_BlV = infinity.get_entropy_logits(
            last_stage[:B],
            cond_BD[:B],
            scale_ind=si,
        ).mul(1 / tau_list[si])

        tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
        logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)  #[bs, vocabulary_size*seq_len, 2]
        B, l, V = logits_BlV.shape
        prob = logits_BlV.softmax(dim=-1).view(-1, V)
        prob_list.append(prob)

        idx_Bld = gt_ls_Bl[si]
        idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)
        idx_Bld = idx_Bld.unsqueeze(1)  # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]
        codes = vae.quantizer.lfq.indices_to_codes(
            idx_Bld, label_type='bit_label')  # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
        if si != final_probability_scale:
            summed_codes += F.interpolate(codes,
                                          size=vae_scale_schedule[-1],
                                          mode=vae.quantizer.z_interplote_up)
            last_stage = F.interpolate(
                summed_codes, size=vae_scale_schedule[si + 1],
                mode=vae.quantizer.z_interplote_up)  # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
            last_stage = last_stage.squeeze(-3)  # [B, d, h, w] or [B, d, 2h, 2w]
            last_stage = last_stage.reshape(*last_stage.shape[:2],
                                            -1)  # [B, d, h*w] or [B, 4d, h*w]
            last_stage = torch.permute(last_stage, [0, 2, 1])  # [B, h*w, d] or [B, h*w, 4d]
        if si != final_probability_scale:
            last_stage = infinity.word_embed(infinity.norm0_ve(last_stage))
            last_stage = last_stage.repeat(bs // B, 1, 1)
    for b in infinity.unregistered_blocks:
        (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)

    return prob_list


@torch.no_grad()
def encoding_to_stream(
    args,
    infinity,
    vae,
    scale_schedule,
    encoder_prompt,
    decoder_prompt,
    text_tokenizer,
    text_encoder,
    gt_ms_idx_Bl,
    *,
    stream_path: str,
    width: int,
    height: int,
    fsync: bool = False,
    max_scale=None,
) -> StreamEncodeResult:
    if max_scale is None:
        final_scale = len(scale_schedule) - 1
    else:
        final_scale = int(max_scale)
        if final_scale < 0 or final_scale >= len(scale_schedule):
            raise ValueError(f"max_scale={max_scale} is outside [0, {len(scale_schedule) - 1}]")

    _sync_model(infinity)
    start = time.perf_counter()
    probabilities = get_prob(
        infinity,
        vae,
        scale_schedule,
        encoder_prompt,
        text_tokenizer,
        text_encoder,
        gt_ms_idx_Bl,
        tau_list=float(getattr(args, "tau", 0.5)),
        cfg_insertion_layer=[0],
        max_scale=final_scale,
    )
    _sync_model(infinity)
    entropy_model_s = time.perf_counter() - start

    start = time.perf_counter()
    scale_payloads: List[bytes] = []
    scale_modes: List[int] = []
    coding_prefix_bits: List[int] = []
    cumulative_bits = 0
    for gt_scale, probability_scale in zip(gt_ms_idx_Bl[:final_scale + 1], probabilities):
        raw_bits = gt_scale.reshape(-1).detach().cpu().numpy().astype(np.uint8)
        raw_payload = np.packbits(raw_bits, bitorder="big").tobytes()
        # Use the exact softmax p(1) consumed by the decoder. Computing
        # 1 - p(0) can differ by one float32 ULP and desynchronize a long
        # arithmetic-coded scale.
        probability_one = probability_scale[:, 1].detach().float().cpu().numpy().astype(np.float32,
                                                                                        copy=False)
        encoder = RangeEncoder()
        encoder.encode(raw_bits, probability_one)
        arithmetic_payload = encoder.finish()
        if len(arithmetic_payload) < len(raw_payload):
            payload = arithmetic_payload
            mode = MODE_ARITHMETIC
        else:
            payload = raw_payload
            mode = MODE_RAW
        scale_payloads.append(payload)
        scale_modes.append(mode)
        cumulative_bits += len(payload) * 8
        coding_prefix_bits.append(int(cumulative_bits))
    range_encode_s = time.perf_counter() - start

    start = time.perf_counter()
    payload, prefix_bytes = build_zero_header_stream(
        width=int(width),
        height=int(height),
        bit_depth=int(args.vae_type),
        scale_schedule=scale_schedule,
        prompt=decoder_prompt,
        scale_payloads=scale_payloads,
        scale_modes=scale_modes,
    )
    serialize_s = time.perf_counter() - start
    start = time.perf_counter()
    write_zero_header_stream(stream_path, payload, fsync=bool(fsync))
    stream_write_s = time.perf_counter() - start
    return StreamEncodeResult(
        stream_path=str(stream_path),
        stream_prefix_bytes=prefix_bytes,
        stream_file_bytes=len(payload),
        coding_prefix_bits=coding_prefix_bits,
        scale_modes=scale_modes,
        timings={
            "entropy_model_s": float(entropy_model_s),
            "range_encode_s": float(range_encode_s),
            "stream_serialize_s": float(serialize_s),
            "stream_write_s": float(stream_write_s),
            "sender_total_s": float(entropy_model_s + range_encode_s + serialize_s +
                                    stream_write_s),
        },
    )
