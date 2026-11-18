"""Scale-wise CSA RangeCoder decoding and CFG prefix reconstruction."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from codec.arithmetic_accel import RangeDecoder
from compression.util import CrossAttnBlock, encode_prompt
from compression.raw_stream_io_v3 import (
    MODE_ARITHMETIC,
    MODE_RAW,
    ZeroHeaderRawStreamContents,
    parse_zero_header_stream,
)

__all__ = ['StreamDecodeResult', 'decoding_from_stream', 'decompress_cfg']


@dataclass
class StreamDecodeResult:
    decoded_indices: List[torch.Tensor]
    prompt: str
    scale_schedule: List[Tuple[int, int, int]]
    stream: ZeroHeaderRawStreamContents
    timings: Dict[str, float]


def _sync_model(model) -> None:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _normalize_schedule(values) -> List[Tuple[int, int, int]]:
    return [tuple(int(item) for item in scale) for scale in values]


def sample_with_top_k_top_p_also_inplace_modifying_logits_(
        logits_BlV: torch.Tensor,
        top_k: int = 0,
        top_p: float = 0.0,
        rng=None,
        num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False,
                                                     dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(
            sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove),
            -torch.inf)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V),
                             num_samples=num_samples,
                             replacement=replacement,
                             generator=rng).view(B, l, num_samples)


def decompress_cfg(
    infinity,
    vae,
    vae_scale_schedule,
    prompt,
    text_tokenizer,
    text_encoder,
    gt_leak,
    gt_ls_Bl,
    cfg_list=3,
    tau_list=0.5,
    cfg_insertion_layer=[0],
):
    rng = infinity.rng
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(vae_scale_schedule)
        cfg_list = [cfg_list] * len(vae_scale_schedule)
    label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, prompt)
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
    B = 1
    if any(np.array(cfg_list) != 1):
        bs = 2 * B
        kv_compact_un = kv_compact.clone()
        total = 0
        for le in lens:
            kv_compact_un[total:total + le] = (infinity.cfg_uncond)[:le]
            total += le
        kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
        cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0)
    else:
        bs = B

    kv_compact = infinity.text_norm(kv_compact)
    sos = cond_BD = infinity.text_proj_for_sos(
        (kv_compact, cu_seqlens_k, max_seqlen_k))  # sos shape: [2, 4096]
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

    accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
    idx_Bl_list, idx_Bld_list = [], []
    num_stages_minus_1 = len(vae_scale_schedule) - 1
    summed_codes = 0
    cur_L = 0
    for si, pn in enumerate(vae_scale_schedule):
        cfg = cfg_list[si]
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

        if (cfg != 1) and add_cfg_on_logits:
            logits_BlV = infinity.get_entropy_logits(
                last_stage,
                cond_BD,
                scale_ind=si,
            ).mul(1 / tau_list[si])
            logits_BlV = cfg * logits_BlV[:B] + (1 - cfg) * logits_BlV[B:]
        else:
            logits_BlV = infinity.get_entropy_logits(
                last_stage[:B],
                cond_BD[:B],
                scale_ind=si,
            ).mul(1 / tau_list[si])

        if infinity.use_bit_label:
            tmp_bs, tmp_seq_len = logits_BlV.shape[:2]  #[bs, seq_len, 64]
            logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)  #[bs, seq_len*32, 2]
            prob = logits_BlV.softmax(dim=-1).view(-1, 2)
            idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV,
                                                                             rng=rng,
                                                                             top_k=0,
                                                                             top_p=0.0,
                                                                             num_samples=1)[:, :, 0]
            idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
        else:
            idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV,
                                                                            rng=rng,
                                                                            top_k=0,
                                                                            top_p=0.0,
                                                                            num_samples=1)[:, :, 0]

        if si <= gt_leak:
            idx_Bld = gt_ls_Bl[si]
        idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)
        idx_Bld = idx_Bld.unsqueeze(1)  # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]
        codes = vae.quantizer.lfq.indices_to_codes(
            idx_Bld, label_type='bit_label')  # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
        if si != num_stages_minus_1:
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
        else:
            summed_codes += codes
        if si != num_stages_minus_1:
            last_stage = infinity.word_embed(infinity.norm0_ve(last_stage))
            last_stage = last_stage.repeat(bs // B, 1, 1)
    for b in infinity.unregistered_blocks:
        (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
    img = vae.decode(summed_codes.squeeze(-3))
    img = (img + 1) / 2
    return img


@torch.no_grad()
def decoding_from_stream(
    args,
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    *,
    stream_path: str,
    expected_scale_schedule=None,
    max_scale: Optional[int] = None,
    restore_rng: bool = True,
) -> StreamDecodeResult:
    receiver_start = time.perf_counter()
    start = time.perf_counter()
    with open(stream_path, "rb") as handle:
        payload = handle.read()
    stream_read_s = time.perf_counter() - start
    start = time.perf_counter()
    stream = parse_zero_header_stream(
        payload,
        max_scale=max_scale,
        bit_depth=int(args.vae_type),
    )
    stream_parse_s = time.perf_counter() - start
    schedule = _normalize_schedule(stream.metadata["scale_schedule"])
    if expected_scale_schedule is not None:
        expected = _normalize_schedule(expected_scale_schedule)
        if schedule != expected:
            raise ValueError(f"zero-header stream schedule mismatch: {schedule} != {expected}")
    if int(args.vae_type) not in (16, 32):
        raise ValueError("zero-header profile requires vae_type=16 or vae_type=32")
    if int(stream.metadata.get("bit_depth", args.vae_type)) != int(args.vae_type):
        raise ValueError(
            f"zero-header bit-depth mismatch: stream={stream.metadata.get('bit_depth')} "
            f"args.vae_type={args.vae_type}")

    prompt = stream.prompt
    entropy_schedule = schedule[:len(stream.scale_payloads)]
    _sync_model(infinity)
    start = time.perf_counter()
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = encode_prompt(text_tokenizer, text_encoder,
                                                                 prompt)
    B = bs = 1
    kv_compact = infinity.text_norm(kv_compact)
    sos = cond_BD = infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
    kv_compact = infinity.text_proj_for_ca(kv_compact)
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + infinity.pos_start.expand(bs, 1, -1)
    with torch.amp.autocast("cuda", enabled=False):
        cond_BD_or_gss = infinity.shared_ada_lin(cond_BD.float()).float().contiguous()
    _sync_model(infinity)
    text_condition_s = time.perf_counter() - start
    for block in infinity.unregistered_blocks:
        (block.sa if isinstance(block, CrossAttnBlock) else block.attn).kv_caching(True)

    _sync_model(infinity)
    start = time.perf_counter()
    decoded_indices: List[torch.Tensor] = []
    summed_codes = 0
    final_decoded_scale_index = len(entropy_schedule) - 1
    for scale_index, pn in enumerate(entropy_schedule):
        attn_fn = None
        if infinity.use_flex_attn:
            attn_fn = infinity.attn_fn_compile_dict.get(tuple(schedule[:scale_index + 1]), None)
        for block_index, chunk in enumerate(infinity.block_chunks):
            if infinity.add_lvl_embeding_only_first_block and block_index == 0:
                last_stage = infinity.add_lvl_embeding(last_stage,
                                                       scale_index,
                                                       schedule,
                                                       need_to_pad=0)
            if not infinity.add_lvl_embeding_only_first_block:
                last_stage = infinity.add_lvl_embeding(last_stage,
                                                       scale_index,
                                                       schedule,
                                                       need_to_pad=0)
            for module in chunk.module:
                last_stage = module(
                    x=last_stage,
                    cond_BD=cond_BD_or_gss,
                    ca_kv=ca_kv,
                    attn_bias_or_two_vector=None,
                    attn_fn=attn_fn,
                    scale_schedule=schedule,
                    rope2d_freqs_grid=infinity.rope2d_freqs_grid,
                    scale_ind=scale_index,
                )
        logits = infinity.get_entropy_logits(last_stage[:B], cond_BD[:B],
                                             scale_ind=scale_index).mul(
                                                 1.0 / float(getattr(args, "tau", 0.5)))
        logits = logits.reshape(logits.shape[0], -1, 2)
        probability_one = (logits.softmax(dim=-1).view(
            -1, 2)[:, 1].detach().float().cpu().numpy()).astype(np.float32, copy=False)
        symbol_count = int(np.array(pn).prod()) * int(args.vae_type)
        scale_payload = stream.scale_payloads[scale_index]
        mode = stream.scale_modes[scale_index]
        if mode == MODE_RAW:
            decompressed = np.unpackbits(np.frombuffer(scale_payload, dtype=np.uint8),
                                         bitorder="big")[:symbol_count]
        elif mode == MODE_ARITHMETIC:
            decompressed = RangeDecoder(scale_payload).decode(probability_one)
        else:
            raise ValueError(f"unsupported zero-header scale mode={mode}")
        if int(decompressed.size) != symbol_count:
            raise ValueError("zero-header decoded symbol count mismatch")

        decoded = torch.from_numpy(np.asarray(decompressed, dtype=np.uint8).copy()).to(
            device=logits.device, dtype=torch.int32).reshape(B,
                                                             int(pn[1]) * int(pn[2]), -1)
        decoded_indices.append(decoded)
        idx_Bld = decoded.reshape(B, int(pn[1]), int(pn[2]), -1).unsqueeze(1)
        codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type="bit_label")
        if scale_index != final_decoded_scale_index:
            summed_codes += F.interpolate(codes,
                                          size=schedule[-1],
                                          mode=vae.quantizer.z_interplote_up)
            last_stage = F.interpolate(
                summed_codes,
                size=schedule[scale_index + 1],
                mode=vae.quantizer.z_interplote_up,
            )
            last_stage = last_stage.squeeze(-3).reshape(*last_stage.shape[:2], -1)
            last_stage = torch.permute(last_stage, [0, 2, 1])
            last_stage = infinity.word_embed(infinity.norm0_ve(last_stage))

    for block in infinity.unregistered_blocks:
        (block.sa if isinstance(block, CrossAttnBlock) else block.attn).kv_caching(False)
    _sync_model(infinity)
    entropy_decode_s = time.perf_counter() - start
    if restore_rng:
        infinity.rng.manual_seed(0)
    receiver_wall_s = time.perf_counter() - receiver_start
    return StreamDecodeResult(
        decoded_indices=decoded_indices,
        prompt=prompt,
        scale_schedule=schedule,
        stream=stream,
        timings={
            "stream_read_s": float(stream_read_s),
            "stream_parse_s": float(stream_parse_s),
            "prompt_decode_s": 0.0,
            "text_condition_s": float(text_condition_s),
            "range_decode_s": float(entropy_decode_s),
            "receiver_shared_total_s": float(stream_read_s + stream_parse_s + text_condition_s +
                                             entropy_decode_s),
            "receiver_wall_s": float(receiver_wall_s),
        },
    )
