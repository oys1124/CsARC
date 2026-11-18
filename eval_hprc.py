"""HPRC evaluation with bit-level allocation and two-pass CFG decoding.

Entropy decoding uses CFG-free probabilities. Reconstruction applies CFG and
restores the transmitted bits.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from compression.prompt_codec import encode_prompt_record
from compression.rc_progressive_codec_bit_level import (
    decode_progressive_rate_control,
    encode_progressive_rate_control,
)
from compression.hprc_core import resolve_target_rate_params
from compression.eval_dataset import load_caption_rows as _load_caption_rows
from compression.rc_progressive_profile import salt_tag
from compression.rc_progressive_stream import (
    pack_progressive_stream,
    pack_progressive_target_stream,
    prompt_mode_name,
    read_progressive_stream,
    read_progressive_target_stream,
    select_progressive_prompt_payload,
    write_progressive_stream,
    write_progressive_target_stream,
)
from compression.prefix_hash_seed import prefix_hash_salt_tag
from compression.rc_progressive_resume import (
    acquire_output_lock,
    open_result_csv,
    read_completed_keys,
    result_key,
    validate_saved_config,
    write_config,
    write_summary,
)
from compression.util import encode_prompt, load_img, mask_quant
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer


DEFAULT_PREFIX_HASH_SALT = "InfinityCodecPrefixSeedV1"
DEFAULT_RATE_HASH_SALT = "InfinityCodecRateControlSeedV1"


def _hash_seed_tag(seed_policy: str, salt: str) -> str:
    if seed_policy == "prefix_hash":
        return prefix_hash_salt_tag(salt)
    if seed_policy == "rate_hash":
        return salt_tag(salt)
    return ""


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)



def _scale_schedule_for_image(img_b3hw: torch.Tensor, pn: str):
    _, _, h, w = img_b3hw.shape
    h_div_w = float(h) / float(w)
    template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[template][pn]["scales"]
    scale_schedule = [(1, int(hh), int(ww)) for _, hh, ww in scale_schedule]
    scale_q = [
        (
            scale_schedule[i][0],
            scale_schedule[i][1],
            scale_schedule[i][2],
            int((i + 1) // ((len(scale_schedule) // 3) + 1) + 2),
        )
        for i in range(len(scale_schedule))
    ]
    return scale_schedule, scale_q


def _prompt_bits_and_decoder_prompt(args, tokenizer, prompt: str):
    prompt_model_bits = 0
    decoder_prompt = prompt
    if args.prompt_bits_mode == "zlib":
        prompt_model_bits = len(__import__("zlib").compress(prompt.encode("utf-8"))) * 8
    elif args.prompt_bits_mode != "none":
        _, prompt_model_bits, decoder_prompt = encode_prompt_record(
            tokenizer, prompt, mode=args.prompt_bits_mode, max_len=int(args.tlen)
        )
    elif int(args.add_prompt_bits):
        _, prompt_model_bits, decoder_prompt = encode_prompt_record(
            tokenizer, prompt, mode=args.prompt_bits_mode, max_len=int(args.tlen)
        )

    prompt_mode, prompt_payload = select_progressive_prompt_payload(str(decoder_prompt))
    prompt_stream_bytes = 2 + len(prompt_payload)
    prompt_stream_bits = prompt_stream_bytes * 8
    return (
        int(prompt_stream_bits),
        str(decoder_prompt),
        int(prompt_model_bits),
        prompt_mode_name(prompt_mode),
        int(prompt_stream_bytes),
    )


def _compute_metrics(source_01: torch.Tensor, recon_bhwc_u8: torch.Tensor, device, lpips_fn, dists_fn):
    recon_01 = (
        recon_bhwc_u8[0]
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=source_01.device, dtype=torch.float32)
        / 255.0
    )
    mse = torch.mean((source_01 - recon_01) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item() if mse.item() > 0 else float("inf")
    msssim_val = ms_ssim(source_01, recon_01, data_range=1.0, size_average=True).item()
    lpips_val = (
        float("nan")
        if lpips_fn is None
        else lpips_fn(source_01.to(device) * 2 - 1, recon_01.to(device) * 2 - 1).item()
    )
    dists_val = (
        float("nan")
        if dists_fn is None
        else dists_fn(source_01.to(device), recon_01.to(device)).item()
    )
    return psnr, msssim_val, lpips_val, dists_val


def _build_model_args(args):
    return argparse.Namespace(
        pn=args.pn,
        model_path=args.model_path,
        cfg_insertion_layer=0,
        vae_type=int(args.vae_type),
        vae_path=args.vae_path,
        add_lvl_embeding_only_first_block=int(args.add_lvl_embeding_only_first_block),
        use_bit_label=1,
        model_type=args.model_type,
        rope2d_each_sa_layer=int(args.rope2d_each_sa_layer),
        rope2d_normalized_by_hw=int(args.rope2d_normalized_by_hw),
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=args.text_encoder_ckpt,
        text_channels=int(args.text_channels),
        apply_spatial_patchify=int(args.apply_spatial_patchify),
        h_div_w_template=1.0,
        use_flex_attn=int(args.use_flex_attn),
        cache_dir=args.cache_dir,
        enable_model_cache=int(args.enable_model_cache),
        checkpoint_type=args.checkpoint_type,
        enable_csa=0,
        seed=int(args.seed),
        bf16=int(args.bf16),
        rec_path="",
    )


def _validate_decode_policy(args) -> None:
    # Two-pass decoding supports CFG values other than one; temperatures and
    # probability/cost scales must still be finite and strictly positive.
    if not np.isfinite(args.decode_cfg):
        raise ValueError("--decode_cfg must be finite")
    for field in ('decode_tau', 'prob_tau', 'budget_rate_scale'):
        value = float(getattr(args, field))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"--{field} must be finite and positive")


def _keep_gpu_busy_loop(
    *,
    args,
    model_args,
    model,
    vae,
    text_tokenizer,
    text_encoder,
    rows_data: List[Dict],
    targets: List[float],
    ms_values: List[str],
    codec_params: Dict,
    device: torch.device,
) -> None:
    if not rows_data or not targets or not ms_values:
        print("[keep_gpu_busy] no image/target/ms is available; skip keep-alive loop.")
        return

    item = rows_data[0]
    img_path = item["img_path"]
    prompt = item["txt"]
    target_bpp = max(float(value) for value in targets)
    ms = str(ms_values[0])
    print(
        "[*] keep_gpu_busy=1, entering GPU keep-alive loop with "
        f"image={os.path.basename(img_path)} ms={ms} target_bpp={target_bpp:g}"
    )

    while True:
        try:
            inp_b3hw = load_img(img_path, model_args).to(device)
            _, _, height, width = inp_b3hw.shape
            fixed_hw = (int(height), int(width))
            scale_schedule, scale_q = _scale_schedule_for_image(inp_b3hw, args.pn)

            raw_features, _, _ = vae.encode_for_raw_features(
                inp_b3hw, scale_schedule=scale_schedule
            )
            _, gt_ms_idx_Bl = mask_quant(vae, scale_q, raw_features, device)
            prompt_bits, decoder_prompt, _, _, _ = _prompt_bits_and_decoder_prompt(
                args, text_tokenizer, prompt
            )
            text_condition = encode_prompt(text_tokenizer, text_encoder, decoder_prompt)

            encode_result = encode_progressive_rate_control(
                model=model,
                vae=vae,
                scale_schedule=scale_schedule,
                text_condition=text_condition,
                prompt=decoder_prompt,
                gt_ms_idx_Bl=gt_ms_idx_Bl,
                target_bpp=target_bpp,
                prompt_bits=prompt_bits,
                ms=ms,
                bit_depth=int(args.vae_type),
                fixed_hw=fixed_hw,
                params=codec_params,
            )
            _ = decode_progressive_rate_control(
                model=model,
                vae=vae,
                scale_schedule=scale_schedule,
                text_condition=text_condition,
                prompt=decoder_prompt,
                payload=encode_result.payload,
                target_bpp=target_bpp,
                prompt_bits=prompt_bits,
                ms=ms,
                bit_depth=int(args.vae_type),
                fixed_hw=fixed_hw,
                params=codec_params,
            )
            _sync(device)
        except Exception:
            time.sleep(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HPRC bit-level evaluation with two-pass CFG decoding.")
    parser.add_argument("--caption_json", default="")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--resume", type=int, choices=[0, 1], default=0,
                        help="Append to the existing CSV and skip completed image/rate pairs.")
    parser.add_argument("--dry_run", type=int, choices=[0, 1], default=0,
                        help="Validate dataset/rate configuration and report pending work without loading models or writing files.")
    parser.add_argument("--target_bpps", default="0.0005,0.0008,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.15,0.2,0.25")
    parser.add_argument("--ms_list", default="entropy")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="When max_images>0, draw a reproducible random subset instead of taking the first images.",
    )
    parser.add_argument("--fallback_prompt", default="a clear photo")
    parser.add_argument("--write_streams", type=int, choices=[0, 1], default=1)
    parser.add_argument("--stream_target_payload_header", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_recon", type=int, choices=[0, 1], default=1)
    parser.add_argument("--fsync", type=int, choices=[0, 1], default=0)
    parser.add_argument("--keep_gpu_busy", type=int, choices=[0, 1], default=0)

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--vae_type", type=int, choices=[32], default=32)
    parser.add_argument("--model_type", default="infinity_2b")
    parser.add_argument("--text_encoder_ckpt", required=True)
    parser.add_argument("--pn", default="1M")
    parser.add_argument("--text_channels", type=int, default=2048)
    parser.add_argument("--apply_spatial_patchify", type=int, choices=[0, 1], default=0)
    parser.add_argument("--add_lvl_embeding_only_first_block", type=int, choices=[0, 1], default=1)
    parser.add_argument("--rope2d_each_sa_layer", type=int, choices=[0, 1], default=1)
    parser.add_argument("--rope2d_normalized_by_hw", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--use_flex_attn", type=int, choices=[0, 1], default=0)
    parser.add_argument("--bf16", type=int, choices=[0, 1], default=1)
    parser.add_argument("--checkpoint_type", default="torch")
    parser.add_argument("--cache_dir", default="/tmp/infinity_rc_cache")
    parser.add_argument("--enable_model_cache", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--add_prompt_bits", type=int, choices=[0, 1], default=1)
    parser.add_argument("--prompt_bits_mode", choices=["arith", "zlib", "none"], default="zlib")
    parser.add_argument("--tlen", type=int, default=512)

    parser.add_argument("--scale_weight", choices=["area", "sqrt", "none"], default="area")
    parser.add_argument("--bit_select_order", choices=["high", "low", "sequential", "raster"], default="high")
    parser.add_argument("--prob_tau", type=float, default=0.5)
    parser.add_argument("--budget_rate_scale", type=float, default=1.6)
    parser.add_argument("--decode_cfg", type=float, default=3.0)
    parser.add_argument("--decode_tau", type=float, default=0.5)
    parser.add_argument("--cfg_insertion_layer", default="0")
    parser.add_argument("--rate_overhead_bits", type=float, default=256.0)
    parser.add_argument(
        "--budget_rate_scale_schedule",
        default="",
        help="piecewise target_lower_bound:value schedule, e.g. 0.0:1.6,0.15:1.65",
    )
    parser.add_argument(
        "--rate_overhead_schedule",
        default="",
        help="piecewise target_lower_bound:value schedule, e.g. 0.0:256,0.15:4096",
    )
    parser.add_argument("--budget_tolerance_bits", type=float, default=1e-6)
    parser.add_argument("--seed_policy", choices=["prefix_hash", "rate_hash", "fixed", "none"], default="prefix_hash")
    parser.add_argument("--hash_seed_salt", default=DEFAULT_RATE_HASH_SALT)
    parser.add_argument("--fixed_seed", type=int, default=0)
    parser.add_argument("--decode_do_sample", type=int, choices=[0, 1], default=1)
    parser.add_argument("--empty_cache_before_vae", type=int, choices=[0, 1], default=1)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    _validate_decode_policy(args)
    stream_dir = os.path.join(args.out_dir, "streams")
    recon_dir = os.path.join(args.out_dir, "recon")

    rows_data = _load_caption_rows(args.caption_json, args.data_dir, args.fallback_prompt)
    if args.max_images > 0:
        sample_size = min(int(args.max_images), len(rows_data))
        if args.sample_seed is None:
            rows_data = rows_data[:sample_size]
        else:
            rows_data = random.Random(int(args.sample_seed)).sample(rows_data, sample_size)
    targets = [float(value.strip()) for value in args.target_bpps.split(",") if value.strip()]
    ms_values = [value.strip().lower() for value in args.ms_list.split(",") if value.strip()]
    for ms in ms_values:
        if ms not in {"rdproxy", "entropy"}:
            raise ValueError(f"unsupported ms={ms!r}; expected rdproxy or entropy")

    codec_params = {
        "scale_weight": args.scale_weight,
        "bit_select_order": args.bit_select_order,
        "prob_tau": float(args.prob_tau),
        "budget_rate_scale": float(args.budget_rate_scale),
        "decode_cfg": float(args.decode_cfg),
        "decode_tau": float(args.decode_tau),
        "cfg_insertion_layer": args.cfg_insertion_layer,
        "rate_overhead_bits": float(args.rate_overhead_bits),
        "budget_rate_scale_schedule": args.budget_rate_scale_schedule,
        "rate_overhead_schedule": args.rate_overhead_schedule,
        "budget_tolerance_bits": float(args.budget_tolerance_bits),
        "seed_policy": args.seed_policy,
        "hash_seed_salt": args.hash_seed_salt,
        "fixed_seed": int(args.fixed_seed),
        "decode_do_sample": bool(args.decode_do_sample),
        "empty_cache_before_vae": bool(args.empty_cache_before_vae),
    }

    csv_path = os.path.join(args.out_dir, "rate_control_progressive.csv")
    fields = [
        "image",
        "ms",
        "target_bpp",
        "bpp",
        "payload_bpp",
        "prompt_bpp",
        "stream_bytes",
        "target_header_bytes",
        "prompt_stream_bytes",
        "prompt_payload_mode",
        "target_error_bpp",
        "payload_bytes",
        "prompt_bits",
        "prompt_model_bits",
        "target_total_bits",
        "target_payload_bits",
        "predicted_symbol_bits",
        "boundary_scale",
        "full_scales",
        "partial_positions",
        "per_scale_kept_pos",
        "partial_bits",
        "per_scale_kept_bits",
        "partial_tokens",
        "full_tokens_in_partial_scale",
        "empty_tokens_in_partial_scale",
        "missing_bit_count",
        "missing_bit_acc",
        "mixed_token_exact_rate",
        "partial_token_exact_rate",
        "partial_token_hamming_mean",
        "partial_token_hamming_hist",
        "code_mse_mixed_gt",
        "code_cosine_mixed_gt",
        "partial_code_mse_mixed_gt",
        "partial_code_cosine_mixed_gt",
        "seed_policy",
        "hash_seed_tag",
        "decode_seed",
        "bit_select_order",
        "prob_tau",
        "budget_rate_scale",
        "rate_overhead_bits",
        "decode_cfg",
        "decode_tau",
        "cfg_insertion_layer",
        "psnr",
        "ms_ssim",
        "lpips",
        "dists",
        "vae_encode_s",
        "mask_quant_s",
        "text_condition_s",
        "encode_core_s",
        "stream_write_s",
        "stream_read_s",
        "decode_core_s",
        "png_write_s",
        "metric_s",
        "encode_total_s",
        "decode_total_s",
    ]
    names = [os.path.basename(item["img_path"]) for item in rows_data]
    if len(set(names)) != len(names):
        raise ValueError("Dataset image basenames must be unique for resume/output paths")
    if not targets or not ms_values or not rows_data:
        raise ValueError("The dataset, target_bpps and ms_list must be nonempty")
    if len(set(targets)) != len(targets) or len(set(ms_values)) != len(ms_values):
        raise ValueError("Duplicate targets or ms values would produce duplicate CSV records")
    expected_keys = {
        result_key(name, ms, target)
        for name in names for ms in ms_values for target in targets
    }
    expected_parameters = {}
    for ms in ms_values:
        for target in targets:
            resolved = resolve_target_rate_params(codec_params, target)
            expected_parameters[(ms, target)] = {
                key: resolved[key] for key in (
                    "bit_select_order", "prob_tau", "budget_rate_scale", "rate_overhead_bits",
                    "decode_cfg", "decode_tau", "cfg_insertion_layer", "seed_policy",
                )
            }
            expected_parameters[(ms, target)].update(
                hash_seed_tag=_hash_seed_tag(args.seed_policy, args.hash_seed_salt),
                target_header_bytes=4 if args.stream_target_payload_header else 0,
            )
    config_path = os.path.join(args.out_dir, "rate_control_progressive.config.json")
    summary_path = os.path.join(args.out_dir, "rate_control_progressive.summary.csv")
    run_lock = None
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
        run_lock = acquire_output_lock(args.out_dir)
    if args.resume:
        validate_saved_config(config_path, vars(args))
        completed = read_completed_keys(csv_path, fields, expected_keys, expected_parameters)
    else:
        completed = set()
    pending = expected_keys - completed
    pending_images = {key[0] for key in pending}
    print(f"[*] Resume={args.resume}: {len(completed)} completed pairs, "
          f"{len(pending)} pending pairs across {len(pending_images)}/{len(rows_data)} images.")
    if args.dry_run:
        return
    write_config(config_path, vars(args))
    if not pending:
        write_summary(csv_path, summary_path)
        print(f"[*] All pairs already complete; summary saved to {summary_path}")
        run_lock.close()
        return
    if args.write_streams:
        os.makedirs(stream_dir, exist_ok=True)
    if args.save_recon:
        os.makedirs(recon_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = _build_model_args(args)
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(model_args)
    model = load_transformer(vae, model_args)

    global ms_ssim, cv2
    import cv2
    from pytorch_msssim import ms_ssim
    import lpips
    from DISTS_pytorch import DISTS

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    dists_fn = DISTS().to(device)

    handle, writer = open_result_csv(csv_path, fields, args.resume)
    with handle:
        for index, item in enumerate(rows_data, start=1):
            img_path = item["img_path"]
            prompt = item["txt"]
            img_name = os.path.basename(img_path)
            stem = os.path.splitext(img_name)[0]
            if img_name not in pending_images:
                continue

            inp_b3hw = load_img(img_path, model_args).to(device)
            _, _, height, width = inp_b3hw.shape
            fixed_hw = (int(height), int(width))
            source_01 = ((inp_b3hw + 1.0) / 2.0).clamp(0.0, 1.0).detach().cpu()
            scale_schedule, scale_q = _scale_schedule_for_image(inp_b3hw, args.pn)

            _sync(device)
            start = time.perf_counter()
            raw_features, _, _ = vae.encode_for_raw_features(inp_b3hw, scale_schedule=scale_schedule)
            _sync(device)
            vae_encode_s = time.perf_counter() - start

            _sync(device)
            start = time.perf_counter()
            _, gt_ms_idx_Bl = mask_quant(vae, scale_q, raw_features, device)
            _sync(device)
            mask_quant_s = time.perf_counter() - start

            (
                prompt_bits,
                decoder_prompt,
                prompt_model_bits,
                prompt_payload_mode,
                prompt_budget_stream_bytes,
            ) = _prompt_bits_and_decoder_prompt(args, text_tokenizer, prompt)
            _sync(device)
            start = time.perf_counter()
            text_condition = encode_prompt(text_tokenizer, text_encoder, decoder_prompt)
            _sync(device)
            text_condition_s = time.perf_counter() - start

            for ms in ms_values:
                for target_bpp in targets:
                    key = result_key(img_name, ms, target_bpp)
                    if key in completed:
                        continue
                    target_codec_params = resolve_target_rate_params(codec_params, target_bpp)
                    encode_result = encode_progressive_rate_control(
                        model=model,
                        vae=vae,
                        scale_schedule=scale_schedule,
                        text_condition=text_condition,
                        prompt=decoder_prompt,
                        gt_ms_idx_Bl=gt_ms_idx_Bl,
                        target_bpp=target_bpp,
                        prompt_bits=prompt_bits,
                        ms=ms,
                        bit_depth=int(args.vae_type),
                        fixed_hw=fixed_hw,
                        params=target_codec_params,
                    )

                    stream_write_s = 0.0
                    stream_read_s = 0.0
                    payload = encode_result.payload
                    if int(args.stream_target_payload_header):
                        stream_bytes = len(
                            pack_progressive_target_stream(
                                prompt=decoder_prompt,
                                payload=payload,
                                target_payload_bits=int(encode_result.target_payload_bits),
                            )
                        )
                        target_header_bytes = 4
                    else:
                        stream_bytes = len(pack_progressive_stream(prompt=decoder_prompt, payload=payload))
                        target_header_bytes = 0
                    prompt_stream_bytes = stream_bytes - len(payload) - target_header_bytes
                    actual_prompt_payload_mode = prompt_payload_mode
                    stream_prompt = decoder_prompt
                    stream_path = ""
                    if args.write_streams:
                        target_tag = f"{target_bpp:.8f}".rstrip("0").rstrip(".").replace(".", "p")
                        stream_path = os.path.join(stream_dir, ms, f"t{target_tag}", stem + ".rcp")
                        start = time.perf_counter()
                        if int(args.stream_target_payload_header):
                            stream_bytes = write_progressive_target_stream(
                                stream_path,
                                prompt=decoder_prompt,
                                payload=payload,
                                target_payload_bits=int(encode_result.target_payload_bits),
                                fsync=bool(args.fsync),
                            )
                        else:
                            stream_bytes = write_progressive_stream(
                                stream_path,
                                prompt=decoder_prompt,
                                payload=payload,
                                fsync=bool(args.fsync),
                            )
                        stream_write_s = time.perf_counter() - start
                        start = time.perf_counter()
                        if int(args.stream_target_payload_header):
                            stream_contents = read_progressive_target_stream(stream_path)
                            if int(stream_contents.target_payload_bits) != int(encode_result.target_payload_bits):
                                raise RuntimeError("target_payload_bits changed after stream roundtrip")
                        else:
                            stream_contents = read_progressive_stream(stream_path)
                        stream_read_s = time.perf_counter() - start
                        payload = stream_contents.payload
                        stream_prompt = stream_contents.prompt
                        stream_bytes = stream_contents.file_bytes
                        prompt_stream_bytes = stream_contents.prompt_bytes
                        target_header_bytes = stream_contents.target_header_bytes
                        actual_prompt_payload_mode = stream_contents.prompt_mode

                    decode_result = decode_progressive_rate_control(
                        model=model,
                        vae=vae,
                        scale_schedule=scale_schedule,
                        text_condition=text_condition,
                        prompt=stream_prompt,
                        payload=payload,
                        target_bpp=target_bpp,
                        prompt_bits=prompt_bits,
                        ms=ms,
                        bit_depth=int(args.vae_type),
                        fixed_hw=fixed_hw,
                        params=target_codec_params,
                        gt_ms_idx_Bl=gt_ms_idx_Bl,
                    )

                    png_write_s = 0.0
                    if args.save_recon:
                        target_tag = f"{target_bpp:.8f}".rstrip("0").rstrip(".").replace(".", "p")
                        out_subdir = os.path.join(recon_dir, ms, f"t{target_tag}")
                        os.makedirs(out_subdir, exist_ok=True)
                        recon_path = os.path.join(out_subdir, img_name)
                        start = time.perf_counter()
                        Image.fromarray(decode_result.image_bhwc_u8[0].detach().cpu().numpy()).save(recon_path)
                        png_write_s = time.perf_counter() - start

                    start = time.perf_counter()
                    psnr, msssim_val, lpips_val, dists_val = _compute_metrics(
                        source_01, decode_result.image_bhwc_u8.detach().cpu(), device, lpips_fn, dists_fn
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    metric_s = time.perf_counter() - start

                    payload_bpp = float(len(payload) * 8) / float(height * width)
                    prompt_bpp = float(prompt_stream_bytes * 8) / float(height * width)
                    total_bpp = float(stream_bytes * 8) / float(height * width)
                    row = {
                        "image": img_name,
                        "ms": ms,
                        "target_bpp": float(target_bpp),
                        "bpp": total_bpp,
                        "payload_bpp": payload_bpp,
                        "prompt_bpp": prompt_bpp,
                        "stream_bytes": int(stream_bytes),
                        "target_header_bytes": int(target_header_bytes),
                        "prompt_stream_bytes": int(prompt_stream_bytes),
                        "prompt_payload_mode": actual_prompt_payload_mode,
                        "target_error_bpp": total_bpp - float(target_bpp),
                        "payload_bytes": int(len(payload)),
                        "prompt_bits": int(prompt_bits),
                        "prompt_model_bits": int(prompt_model_bits),
                        "target_total_bits": int(encode_result.target_total_bits),
                        "target_payload_bits": int(encode_result.target_payload_bits),
                        "predicted_symbol_bits": float(encode_result.predicted_symbol_bits),
                        "boundary_scale": int(encode_result.boundary_scale),
                        "full_scales": int(encode_result.full_scales),
                        "partial_positions": int(encode_result.partial_positions),
                        "per_scale_kept_pos": json.dumps(encode_result.per_scale_kept_pos, separators=(",", ":")),
                        "partial_bits": int(encode_result.partial_bits),
                        "per_scale_kept_bits": json.dumps(encode_result.per_scale_kept_bits, separators=(",", ":")),
                        "partial_tokens": int(decode_result.partial_tokens),
                        "full_tokens_in_partial_scale": int(decode_result.full_tokens_in_partial_scale),
                        "empty_tokens_in_partial_scale": int(decode_result.empty_tokens_in_partial_scale),
                        "missing_bit_count": int(decode_result.missing_bit_count),
                        "missing_bit_acc": float(decode_result.missing_bit_acc),
                        "mixed_token_exact_rate": float(decode_result.mixed_token_exact_rate),
                        "partial_token_exact_rate": float(decode_result.partial_token_exact_rate),
                        "partial_token_hamming_mean": float(decode_result.partial_token_hamming_mean),
                        "partial_token_hamming_hist": json.dumps(
                            decode_result.partial_token_hamming_hist, separators=(",", ":")
                        ),
                        "code_mse_mixed_gt": float(decode_result.code_mse_mixed_gt),
                        "code_cosine_mixed_gt": float(decode_result.code_cosine_mixed_gt),
                        "partial_code_mse_mixed_gt": float(decode_result.partial_code_mse_mixed_gt),
                        "partial_code_cosine_mixed_gt": float(decode_result.partial_code_cosine_mixed_gt),
                        "seed_policy": args.seed_policy,
                        "hash_seed_tag": _hash_seed_tag(args.seed_policy, args.hash_seed_salt),
                        "decode_seed": "" if encode_result.decode_seed is None else int(encode_result.decode_seed),
                        "bit_select_order": args.bit_select_order,
                        "prob_tau": float(target_codec_params["prob_tau"]),
                        "budget_rate_scale": float(target_codec_params["budget_rate_scale"]),
                        "rate_overhead_bits": float(target_codec_params["rate_overhead_bits"]),
                        "decode_cfg": float(args.decode_cfg),
                        "decode_tau": float(args.decode_tau),
                        "cfg_insertion_layer": str(args.cfg_insertion_layer),
                        "psnr": psnr,
                        "ms_ssim": msssim_val,
                        "lpips": lpips_val,
                        "dists": dists_val,
                        "vae_encode_s": vae_encode_s,
                        "mask_quant_s": mask_quant_s,
                        "text_condition_s": text_condition_s,
                        "encode_core_s": encode_result.timings["encode_core_s"],
                        "stream_write_s": stream_write_s,
                        "stream_read_s": stream_read_s,
                        "decode_core_s": decode_result.timings["decode_core_s"],
                        "png_write_s": png_write_s,
                        "metric_s": metric_s,
                        "encode_total_s": vae_encode_s
                        + mask_quant_s
                        + text_condition_s
                        + encode_result.timings["encode_core_s"]
                        + stream_write_s,
                        "decode_total_s": text_condition_s
                        + stream_read_s
                        + decode_result.timings["decode_core_s"],
                    }
                    writer.writerow(row)
                    handle.flush()
                    if args.fsync:
                        os.fsync(handle.fileno())
                    completed.add(key)
                    print(
                        f"[{index}/{len(rows_data)}] {img_name} ms={ms} target={target_bpp:g} "
                        f"bpp={total_bpp:.6f} boundary={row['boundary_scale']} "
                        f"partial_bits={row['partial_bits']} partial_tokens={row['partial_tokens']} "
                        f"miss_acc={row['missing_bit_acc']:.4f} partial_exact={row['partial_token_exact_rate']:.4f} "
                        f"PSNR={psnr:.4f} MS-SSIM={msssim_val:.6f}"
                    )

    if completed != expected_keys:
        raise RuntimeError(f"Evaluation still has {len(expected_keys - completed)} missing pairs")
    write_summary(csv_path, summary_path)
    write_config(config_path, vars(args))
    print(f"[*] CSV saved to {csv_path}")
    print(f"[*] Summary saved to {summary_path}")
    print(f"[*] Config saved to {config_path}")
    run_lock.close()
    if int(args.keep_gpu_busy) == 1:
        _keep_gpu_busy_loop(
            args=args,
            model_args=model_args,
            model=model,
            vae=vae,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
            rows_data=rows_data,
            targets=targets,
            ms_values=ms_values,
            codec_params=codec_params,
            device=device,
        )


if __name__ == "__main__":
    main()
