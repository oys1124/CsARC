"""Scale-wise image compression and reconstruction evaluation."""
import os
import sys
import torch
import torchvision
import tempfile
import time
import re
import json
import argparse
import hashlib
import numpy as np

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from tools.run_infinity import *
from compression.util import *
from compression.sender import encoding_to_stream
from compression.reciever import decompress_cfg, decoding_from_stream
from compression.raw_stream_io_v3 import FIXED_STREAM_DECODE_SEED
from compression.raw_stream_io_v3 import select_zero_header_prompt_payload
from compression.prefix_hash_seed import derive_prefix_hash_seeds, prefix_hash_salt_tag


def _cuda_sync_for(model):
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        return
    if model_device.type == 'cuda':
        torch.cuda.synchronize(model_device)


def _csa_runtime_config(args, infinity=None):
    return {'method': 'CSA', 'auxiliary_inference': False, 'logit_fusion': False}


def _validate_csa_runtime(infinity, args):
    if bool(getattr(infinity, 'enable_csa', False)):
        raise RuntimeError('CSA auxiliary modules must be absent during codec inference')
    print('[*] CSA backbone loaded; auxiliary inference is disabled')


# Normalize the decoder caption through T5 token IDs.
def _encode_prompt_ids_t5(tokenizer, prompt: str, max_len: int = 512):
    enc = tokenizer(prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=True)
    ids = enc['input_ids'][0].tolist()
    return list(map(int, ids))


def _decode_prompt_text_from_ids_t5(tokenizer, ids):
    """
    最简单接法：恢复字符串，再喂 RangeCoder 解码及 CFG 重建。
    """
    if ids is None or len(ids) == 0:
        return ''
    try:
        return tokenizer.decode(ids, skip_special_tokens=True).strip()
    except TypeError:
        return tokenizer.decode(ids)


def _recon_tensor_for_metrics(img: torch.Tensor) -> torch.Tensor:
    grid = torchvision.utils.make_grid(img.detach().float().cpu())
    grid_u8 = grid.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
    return grid_u8.float().div(255.0).unsqueeze(0)


def _pure_generation_seed(args, prompt: str) -> int:
    rng_mode = str(getattr(args, 'rng_mode', 'prefix_hash'))
    if rng_mode == 'prefix_hash':
        hasher = hashlib.blake2b(digest_size=8, person=b'InfPureGenV1')
        hasher.update(str(getattr(args, 'hash_seed_salt', '')).encode('utf-8'))
        hasher.update(b'\x00')
        hasher.update(str(prompt).encode('utf-8'))
        return int.from_bytes(hasher.digest(), 'little') & (1 << 63) - 1
    if rng_mode == 'stream_seed':
        return int(getattr(args, 'decode_seed_base', 0))
    return int(FIXED_STREAM_DECODE_SEED)


def compress_to_file(args,
                     vae,
                     scale_schedule,
                     scale_q,
                     infinity,
                     text_tokenizer,
                     text_encoder,
                     img_path,
                     text,
                     stream_path,
                     decoder_prompt,
                     h,
                     w,
                     max_scale=None):
    timings = {}
    inp_B3HW = load_img(img_path, args)
    _cuda_sync_for(vae)
    start = time.perf_counter()
    (raw_features, _, _) = vae.encode_for_raw_features(inp_B3HW.to(device),
                                                       scale_schedule=scale_schedule)
    _cuda_sync_for(vae)
    timings['vae_encode_s'] = time.perf_counter() - start
    _cuda_sync_for(vae)
    start = time.perf_counter()
    (_, gt_ms_idx_Bl) = mask_quant(vae, scale_q, raw_features, device)
    _cuda_sync_for(vae)
    timings['mask_quant_s'] = time.perf_counter() - start
    encoded = encoding_to_stream(args,
                                 infinity,
                                 vae,
                                 scale_schedule,
                                 text,
                                 decoder_prompt,
                                 text_tokenizer,
                                 text_encoder,
                                 gt_ms_idx_Bl,
                                 stream_path=stream_path,
                                 width=int(w),
                                 height=int(h),
                                 fsync=bool(args.stream_fsync),
                                 max_scale=max_scale)
    timings.update(encoded.timings)
    timings['encode_total_s'] = float(timings['vae_encode_s'] + timings['mask_quant_s'] +
                                      timings['sender_total_s'])
    return (encoded, gt_ms_idx_Bl, timings)





def evaluate_pure_generation(args, infinity, vae, scale_schedule, text, text_tokenizer,
                             text_encoder, rec_base_path, img_path, img_name, device, h, w,
                             prompt_bpp, lpips_fn, dists_fn):
    decode_seed = _pure_generation_seed(args, text)
    infinity.rng.manual_seed(int(decode_seed))
    std_img_cv = cv2.imread(img_path)
    std_img_cv = cv2.cvtColor(std_img_cv, cv2.COLOR_BGR2RGB)
    std_tensor = torch.tensor(std_img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    _cuda_sync_for(infinity)
    start = time.perf_counter()
    img = decompress_cfg(infinity,
                         vae,
                         scale_schedule,
                         text,
                         text_tokenizer,
                         text_encoder,
                         -1, [],
                         cfg_list=float(args.decode_cfg),
                         tau_list=float(args.decode_tau),
                         cfg_insertion_layer=[0])
    _cuda_sync_for(infinity)
    reconstruct_s = time.perf_counter() - start
    if int(getattr(args, 'save_recon', 1)):
        scale_folder = os.path.join(rec_base_path, 'scale_-1')
        os.makedirs(scale_folder, exist_ok=True)
        rec_img_path = os.path.join(scale_folder, img_name)
        start = time.perf_counter()
        torchvision.utils.save_image(img.cpu(), rec_img_path)
        png_write_s = time.perf_counter() - start
        start = time.perf_counter()
        rec_img_cv = cv2.imread(rec_img_path)
        rec_img_cv = cv2.cvtColor(rec_img_cv, cv2.COLOR_BGR2RGB)
        rec_tensor = torch.tensor(rec_img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    else:
        png_write_s = 0.0
        start = time.perf_counter()
        rec_tensor = _recon_tensor_for_metrics(img)
    mse = torch.mean((std_tensor - rec_tensor)**2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item() if mse.item() > 0 else float('inf')
    msssim_val = ms_ssim(std_tensor, rec_tensor, data_range=1.0, size_average=True).item()
    lpips_val = lpips_fn(std_tensor.to(device) * 2 - 1, rec_tensor.to(device) * 2 - 1).item()
    dists_val = dists_fn(std_tensor.to(device), rec_tensor.to(device)).item()
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    metric_s = time.perf_counter() - start
    prompt_bits = int(getattr(args, 'prompt_bits', 0)) if int(getattr(args, 'add_prompt_bits',
                                                                      1)) else 0
    prompt_bytes = int((prompt_bits + 7) // 8)
    return {
        '-1': {
            'bpp': float(prompt_bpp),
            'stream_bpp': float(prompt_bpp),
            'coding_payload_bpp': 0.0,
            'image_payload_stream_bpp': 0.0,
            'standalone_equivalent_bpp': float(prompt_bpp),
            'stream_prefix_bytes': prompt_bytes,
            'rng_mode': str(getattr(args, 'rng_mode', 'prefix_hash')),
            'decode_seed': int(decode_seed),
            'hash_seed_salt': str(getattr(args, 'hash_seed_salt', '')) if str(
                getattr(args, 'rng_mode', 'prefix_hash')) == 'prefix_hash' else '',
            'psnr': psnr,
            'msssim': msssim_val,
            'lpips': lpips_val,
            'dists': dists_val,
            'reconstruct_s': float(reconstruct_s),
            'png_write_s': float(png_write_s),
            'metric_s': float(metric_s)
        }
    }


def decompress_modified_from_file(args,
                                  infinity,
                                  vae,
                                  scale_schedule,
                                  text_tokenizer,
                                  text_encoder,
                                  rec_base_path,
                                  img_path,
                                  img_name,
                                  stream_path,
                                  device,
                                  h,
                                  w,
                                  lpips_fn,
                                  dists_fn,
                                  *,
                                  expected_prompt=None,
                                  expected_gt=None,
                                  max_scale=None,
                                  only_scale=None):
    decoded = decoding_from_stream(args,
                                   infinity,
                                   vae,
                                   text_tokenizer,
                                   text_encoder,
                                   stream_path=stream_path,
                                   expected_scale_schedule=scale_schedule,
                                   max_scale=max_scale,
                                   restore_rng=True)
    if expected_prompt is not None and decoded.prompt != expected_prompt:
        raise AssertionError(
            f'stream prompt mismatch: decoded={decoded.prompt!r}, expected={expected_prompt!r}')
    if expected_gt is not None:
        if len(decoded.decoded_indices) > len(expected_gt):
            raise AssertionError('stream decoded scale count exceeds encoder GT')
        for (scale_index, (actual, expected)) in enumerate(
                zip(decoded.decoded_indices, expected_gt[:len(decoded.decoded_indices)])):
            if not torch.equal(actual.to(expected.device), expected.to(actual.dtype)):
                raise AssertionError(f'stream entropy decode mismatch at scale {scale_index}')
    coding_prefix_bits = []
    coding_total = 0
    payload_bit_counts = [len(payload) * 8 for payload in decoded.stream.scale_payloads]
    for payload_bits in payload_bit_counts:
        coding_total += int(payload_bits)
        coding_prefix_bits.append(coding_total)
    std_img_cv = cv2.imread(img_path)
    std_img_cv = cv2.cvtColor(std_img_cv, cv2.COLOR_BGR2RGB)
    std_tensor = torch.tensor(std_img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    scale_results = {}
    rng_mode = str(args.rng_mode)
    decode_seed = int(args.decode_seed_base)
    prefix_hash_seeds = []
    seed_start = time.perf_counter()
    if rng_mode == 'prefix_hash':
        prefix_hash_seeds = derive_prefix_hash_seeds(decoded.decoded_indices, decoded.prompt,
                                                     args.hash_seed_salt)
    decoded.timings['prefix_hash_seed_s'] = float(time.perf_counter() - seed_start)
    if only_scale is None:
        scale_indices = list(range(len(decoded.decoded_indices)))
    else:
        target_scale = int(only_scale)
        if target_scale < 0 or target_scale >= len(decoded.decoded_indices):
            raise ValueError(
                f'only_scale={only_scale} is unavailable in decoded prefix with {len(decoded.decoded_indices)} scales'
            )
        scale_indices = [target_scale]
    for gt_leak in scale_indices:
        coding_payload_bpp = coding_prefix_bits[gt_leak] / float(h * w)
        stream_bpp = decoded.stream.prefix_bytes[gt_leak] * 8.0 / float(h * w)
        image_payload_stream_bpp = (decoded.stream.prefix_bytes[gt_leak] -
                                    decoded.stream.header_bytes) * 8.0 / float(h * w)
        standalone_equivalent_bpp = stream_bpp

        scale_decode_seed = decode_seed
        if rng_mode == 'prefix_hash':
            scale_decode_seed = int(prefix_hash_seeds[gt_leak])
            infinity.rng.manual_seed(scale_decode_seed)
        elif rng_mode == 'stream_seed':
            infinity.rng.manual_seed(scale_decode_seed)
        _cuda_sync_for(infinity)
        start = time.perf_counter()
        img = decompress_cfg(infinity,
                             vae,
                             scale_schedule,
                             decoded.prompt,
                             text_tokenizer,
                             text_encoder,
                             gt_leak,
                             decoded.decoded_indices,
                             cfg_list=float(args.decode_cfg),
                             tau_list=float(args.decode_tau),
                             cfg_insertion_layer=list(
                                 decoded.stream.metadata.get('cfg_insertion_layer', [0])))
        _cuda_sync_for(infinity)
        reconstruct_s = time.perf_counter() - start
        if int(getattr(args, 'save_recon', 1)):
            scale_folder = os.path.join(rec_base_path, f'scale_{gt_leak}')
            os.makedirs(scale_folder, exist_ok=True)
            rec_img_path = os.path.join(scale_folder, img_name)
            start = time.perf_counter()
            torchvision.utils.save_image(img.cpu(), rec_img_path)
            png_write_s = time.perf_counter() - start
            start = time.perf_counter()
            rec_img_cv = cv2.imread(rec_img_path)
            rec_img_cv = cv2.cvtColor(rec_img_cv, cv2.COLOR_BGR2RGB)
            rec_tensor = torch.tensor(rec_img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        else:
            png_write_s = 0.0
            start = time.perf_counter()
            rec_tensor = _recon_tensor_for_metrics(img)
        mse = torch.mean((std_tensor - rec_tensor)**2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item() if mse.item() > 0 else float('inf')
        msssim_val = ms_ssim(std_tensor, rec_tensor, data_range=1.0, size_average=True).item()
        lpips_val = lpips_fn(std_tensor.to(device) * 2 - 1, rec_tensor.to(device) * 2 - 1).item()
        dists_val = dists_fn(std_tensor.to(device), rec_tensor.to(device)).item()
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        metric_s = time.perf_counter() - start
        scale_results[str(gt_leak)] = {
            'bpp': stream_bpp,
            'stream_bpp': stream_bpp,
            'coding_payload_bpp': coding_payload_bpp,
            'image_payload_stream_bpp': image_payload_stream_bpp,
            'standalone_equivalent_bpp': standalone_equivalent_bpp,
            'stream_prefix_bytes': int(decoded.stream.prefix_bytes[gt_leak]),
            'rng_mode': rng_mode,
            'decode_seed': scale_decode_seed,
            'hash_seed_salt': args.hash_seed_salt if rng_mode == 'prefix_hash' else '',
            'psnr': psnr,
            'msssim': msssim_val,
            'lpips': lpips_val,
            'dists': dists_val,
            'reconstruct_s': float(reconstruct_s),
            'png_write_s': float(png_write_s),
            'metric_s': float(metric_s)
        }
    return (scale_results, decoded.timings)


@torch.no_grad()
def _keep_gpu_busy(args, infinity, vae, text_tokenizer, text_encoder, item):
    """Repeat encoding, file decoding and reconstruction."""
    img_path, prompt = item['img_path'], item['txt']
    image = load_img(img_path, args)
    _, _, height, width = image.shape
    ratio = height / width
    template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - ratio))]
    schedule = [(1, h, w) for _, h, w in dynamic_resolution_h_w[template][args.pn]['scales']]
    scale_q = [(*scale, int((i + 1) // ((len(schedule) // 3) + 1) + 2))
               for i, scale in enumerate(schedule)]
    ids = _encode_prompt_ids_t5(text_tokenizer, prompt, max_len=args.tlen)
    decoder_prompt = _decode_prompt_text_from_ids_t5(text_tokenizer, ids)
    with tempfile.TemporaryDirectory(prefix='csa-rangecoder-keepalive-') as tmp:
        path = os.path.join(tmp, 'image.iraw')
        while True:
            try:
                with torch.autocast('cuda', enabled=torch.cuda.is_available(),
                                    dtype=torch.bfloat16):
                    compress_to_file(args, vae, schedule, scale_q, infinity, text_tokenizer,
                                     text_encoder, img_path, prompt, path, decoder_prompt, height,
                                     width)
                    decoded = decoding_from_stream(args,
                                                   infinity,
                                                   vae,
                                                   text_tokenizer,
                                                   text_encoder,
                                                   stream_path=path,
                                                   expected_scale_schedule=schedule)
                    for scale in range(len(decoded.decoded_indices)):
                        decompress_cfg(infinity,
                                       vae,
                                       schedule,
                                       decoded.prompt,
                                       text_tokenizer,
                                       text_encoder,
                                       scale,
                                       decoded.decoded_indices,
                                       cfg_list=args.decode_cfg,
                                       tau_list=args.decode_tau)
            except Exception as error:
                print(f'[keep_gpu_busy] {error}')
                time.sleep(1)


def build_parser():
    runtime_parser = argparse.ArgumentParser(
        description='Scale-wise file-backed evaluation for the CSA backbone.')
    runtime_parser.add_argument('--verify_stream',
                                type=int,
                                choices=[0, 1],
                                default=0,
                                help='Verify entropy-decoded bits against encoder ground truth.')
    runtime_parser.add_argument(
        '--rng_mode',
        choices=['stream_seed', 'prefix_hash'],
        default='prefix_hash',
        help='prefix_hash derives seeds from transmitted scales; stream_seed uses fixed seed 0.')
    runtime_parser.add_argument(
        '--hash_seed_salt',
        default=os.environ.get('HASH_SEED_SALT', 'InfinityCodecPrefixSeedV1'),
        help=
        'Shared codec-profile salt for deterministic prefix_hash seeds; it is not written to each stream.'
    )
    runtime_parser.add_argument(
        '--decode_seed_base',
        type=int,
        default=0,
        help='Fixed sampling seed; stream_seed requires 0.')
    runtime_parser.add_argument('--stream_root',
                                default='',
                                help='Stream output root. Empty uses <rec_path>/streams.')
    runtime_parser.add_argument('--stream_fsync', type=int, choices=[0, 1], default=0)
    runtime_parser.add_argument(
        '--codec_timing_mode',
        choices=['multirate', 'prefix_only'],
        default='multirate',
        help=
        'multirate encodes one full 13-scale stream and reports shared timing; prefix_only independently encodes/decodes each target scale prefix for strict single-rate timing.'
    )
    runtime_parser.add_argument('--max_images',
                                type=int,
                                default=0,
                                help='0 evaluates all configured images.')
    runtime_parser.add_argument(
        '--tau',
        type=float,
        default=1.0,
        help=
        'Entropy-coding temperature; default 1.0, independent of reconstruction sampling.'
    )
    runtime_parser.add_argument(
        '--decode_tau',
        type=float,
        default=0.5,
        help='Reconstruction sampling temperature; default 0.5.')
    runtime_parser.add_argument(
        '--decode_cfg',
        type=float,
        default=3.0,
        help=
        'Reconstruction classifier-free guidance scale; default 3.0.')
    runtime_parser.add_argument('--keep_gpu_busy', type=int, choices=[0, 1], default=0)
    runtime_parser.add_argument(
        '--save_recon',
        type=int,
        choices=[0, 1],
        default=int(os.environ.get('EVAL_SAVE_RECON', '1')),
        help='1 writes reconstructed PNGs; 0 computes metrics in memory without saving PNGs.')
    runtime_parser.add_argument(
        '--pure_gen_only',
        type=int,
        choices=[0, 1],
        default=int(os.environ.get('EVAL_PURE_GEN_ONLY', '0')),
        help='1 evaluates only scale_-1 pure generation with no transmitted image scales.')
    runtime_parser.add_argument('--model_path', required=True)
    runtime_parser.add_argument('--vae_path', required=True)
    runtime_parser.add_argument('--text_encoder_ckpt', required=True)
    runtime_parser.add_argument('--caption_json', default='')
    runtime_parser.add_argument('--data_dir', default='')
    runtime_parser.add_argument('--fallback_prompt', default='a clear photo')
    runtime_parser.add_argument('--out_dir', default='outputs/csa_scale_eval')
    runtime_parser.add_argument('--vae_type', type=int, choices=[32], default=32)
    runtime_parser.add_argument('--model_type', choices=['infinity_2b'], default='infinity_2b')
    runtime_parser.add_argument('--use_flex_attn', type=int, choices=[0, 1], default=0)
    runtime_parser.add_argument('--bf16', type=int, choices=[0, 1], default=0)
    runtime_parser.add_argument('--sweep_checkpoints', action='store_true')
    runtime_parser.add_argument('--dry_run', action='store_true')
    return runtime_parser


def main(argv=None):
    global device, ms_ssim, cv2, pd
    runtime_args = build_parser().parse_args(argv)
    if float(runtime_args.tau) <= 0:
        raise ValueError(f'--tau must be positive, got {runtime_args.tau}')
    if float(runtime_args.decode_tau) <= 0:
        raise ValueError(f'--decode_tau must be positive, got {runtime_args.decode_tau}')
    if runtime_args.rng_mode == 'stream_seed' and int(runtime_args.decode_seed_base) != 0:
        raise ValueError('stream_seed uses the protocol-fixed seed 0; --decode_seed_base must be 0')
    initial_model_path = runtime_args.model_path
    eval_rec_root = runtime_args.out_dir
    eval_vae_type = runtime_args.vae_type
    vae_path = runtime_args.vae_path
    text_encoder_ckpt = runtime_args.text_encoder_ckpt
    args = argparse.Namespace(pn='1M',
                              model_path=initial_model_path,
                              cfg_insertion_layer=0,
                              vae_type=eval_vae_type,
                              vae_path=vae_path,
                              add_lvl_embeding_only_first_block=1,
                              use_bit_label=1,
                              model_type=runtime_args.model_type,
                              rope2d_each_sa_layer=1,
                              rope2d_normalized_by_hw=2,
                              use_scale_schedule_embedding=0,
                              sampling_per_bits=1,
                              text_encoder_ckpt=text_encoder_ckpt,
                              text_channels=2048,
                              apply_spatial_patchify=0,
                              h_div_w_template=1.0,
                              use_flex_attn=runtime_args.use_flex_attn,
                              cache_dir=os.path.join(eval_rec_root, 'cache'),
                              enable_model_cache=0,
                              checkpoint_type='torch',
                              enable_csa=0,
                              csa_logit_fusion=0,
                              seed=0,
                              bf16=runtime_args.bf16,
                              rec_path='',
                              add_prompt_bits=1,
                              tlen=512,
                              keep_gpu_busy=int(runtime_args.keep_gpu_busy),
                              save_recon=int(runtime_args.save_recon),
                              pure_gen_only=int(runtime_args.pure_gen_only),
                              stream_root=str(runtime_args.stream_root),
                              stream_fsync=int(runtime_args.stream_fsync),
                              codec_timing_mode=str(runtime_args.codec_timing_mode),
                              max_images=int(runtime_args.max_images),
                              rng_mode=str(runtime_args.rng_mode),
                              hash_seed_salt=str(runtime_args.hash_seed_salt),
                              decode_seed_base=int(runtime_args.decode_seed_base),
                              tau=float(runtime_args.tau),
                              decode_tau=float(runtime_args.decode_tau),
                              decode_cfg=float(runtime_args.decode_cfg),
                              current_decode_seed=0,
                              prompt_bits=0,
                              verify_stream=bool(runtime_args.verify_stream))
    from compression.eval_dataset import load_caption_rows as _load_caption_rows
    if runtime_args.dry_run:
        rows = _load_caption_rows(runtime_args.caption_json, runtime_args.data_dir,
                                  runtime_args.fallback_prompt)
        print(
            json.dumps(
                {
                    'method': 'CSA scale-wise',
                    'images': len(rows),
                    'model_path': args.model_path,
                    'stream_format': 'zero_header_v3',
                    'coding_tau': args.tau,
                    'decode_cfg': args.decode_cfg,
                    'decode_tau': args.decode_tau,
                    'auxiliary_inference': False
                },
                indent=2))
        return
    import cv2
    import pandas as pd
    import lpips
    from DISTS_pytorch import DISTS
    from pytorch_msssim import ms_ssim
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    (text_tokenizer, text_encoder) = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    lpips_evaluator = lpips.LPIPS(net='alex').to(device)
    dists_evaluator = DISTS().to(device)
    json_data = []
    from compression.eval_dataset import load_caption_rows as _load_caption_rows
    json_data = _load_caption_rows(runtime_args.caption_json, runtime_args.data_dir,
                                   runtime_args.fallback_prompt)
    if args.max_images > 0:
        json_data = json_data[:args.max_images]
    model_dir = os.path.dirname(initial_model_path)
    base_filename = os.path.basename(initial_model_path)
    match_init = re.search('-iter(\\d+)-', base_filename)
    if match_init and runtime_args.sweep_checkpoints:
        start_iter = int(match_init.group(1))
        available_models = []
        for f in os.listdir(model_dir):
            if f.endswith('.pth') and 'iter' in f:
                m = re.search('-iter(\\d+)-', f)
                if m:
                    available_models.append((int(m.group(1)), os.path.join(model_dir, f), f))
        available_models.sort(key=lambda x: x[0])
        models_to_test = [item for item in available_models if item[0] >= start_iter]
        print(f'[*] 成功扫描到 {len(models_to_test)} 个待测试模型。')
    else:
        single_iter = int(os.environ.get('EVAL_MODEL_ITER', '0'))
        models_to_test = [(single_iter, initial_model_path, base_filename)]
        print(
            f"[*] initial_model_path 不含 '-iterXXXX-'，按单 checkpoint 测试: {base_filename} (Iter: {single_iter})"
        )
    for (current_iter, current_model_path, current_filename) in models_to_test:
        print(f"\n{'=' * 60}")
        print(f'[*] 开始测试模型: {current_filename} (Iter: {current_iter})')
        print(f"{'=' * 60}")
        args.model_path = current_model_path
        base_rec_path = os.path.join(eval_rec_root, f'DIV2K_ori_{current_iter}')
        result_suffix = ''
        if args.enable_csa:
            result_suffix += f'_csa1_fusion{int(bool(args.csa_logit_fusion))}'
        if args.rng_mode == 'stream_seed':
            result_suffix += '_stream_seed'
        elif args.rng_mode == 'prefix_hash':
            result_suffix += f'_prefix_hash_{prefix_hash_salt_tag(args.hash_seed_salt)}'
        result_suffix += '_v3_standalone_zlib'
        if args.codec_timing_mode == 'prefix_only':
            result_suffix += '_prefixonlytiming'
        if int(getattr(args, 'pure_gen_only', 0)):
            result_suffix += '_puregen'
        args.rec_path = base_rec_path + result_suffix
        if not os.path.exists(args.rec_path):
            os.makedirs(args.rec_path)
        infinity = load_transformer(vae, args)
        _validate_csa_runtime(infinity, args)
        csa_config = _csa_runtime_config(args, infinity)
        csa_profile_json = json.dumps(csa_config, sort_keys=True, separators=(',', ':'))
        dataset_metrics_data = []
        codec_timing_rows = []
        for data in json_data:
            (img_path, text) = (data['img_path'], data['txt'])
            img_name = img_path.split('/')[-1]
            args.current_decode_seed = int(FIXED_STREAM_DECODE_SEED)
            inp_B3HW = load_img(img_path, args)
            (b, c, h, w) = inp_B3HW.shape
            h_div_w = h / w
            h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
            scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
            scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
            scale_q = [(scale_schedule[i][0], scale_schedule[i][1], scale_schedule[i][2],
                        int((i + 1) // (len(scale_schedule) // 3 + 1) + 2))
                       for i in range(len(scale_schedule))]
            prompt_ids = _encode_prompt_ids_t5(text_tokenizer, text, max_len=args.tlen)
            text_for_decode = _decode_prompt_text_from_ids_t5(text_tokenizer, prompt_ids)
            prompt_bits = (2 + len(select_zero_header_prompt_payload(text_for_decode)[1])) * 8
            args.prompt_bits = int(prompt_bits)
            prompt_bpp = prompt_bits / (h * w)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if int(getattr(args, 'pure_gen_only', 0)):
                    scale_metrics = evaluate_pure_generation(args, infinity, vae, scale_schedule,
                                                             text_for_decode, text_tokenizer,
                                                             text_encoder, args.rec_path, img_path,
                                                             img_name, device, h, w, prompt_bpp,
                                                             lpips_evaluator, dists_evaluator)
                else:
                    stream_root = os.path.join(
                        args.stream_root,
                        f'iter_{current_iter}') if args.stream_root else os.path.join(
                            args.rec_path, 'streams')
                    os.makedirs(stream_root, exist_ok=True)
                    stream_path = os.path.join(stream_root, os.path.splitext(img_name)[0] + '.iraw')
                    if args.codec_timing_mode == 'prefix_only':
                        scale_metrics = {}
                        stream_stem = os.path.splitext(img_name)[0]
                        for target_scale in range(len(scale_schedule)):
                            prefix_stream_path = os.path.join(
                                stream_root, f'{stream_stem}_s{target_scale:02d}.iraw')
                            (encoded, gt_ms_idx_Bl,
                             encode_timings) = compress_to_file(args,
                                                                vae,
                                                                scale_schedule,
                                                                scale_q,
                                                                infinity,
                                                                text_tokenizer,
                                                                text_encoder,
                                                                img_path,
                                                                text,
                                                                prefix_stream_path,
                                                                text_for_decode,
                                                                h,
                                                                w,
                                                                max_scale=target_scale)
                            print(
                                f"Processing {img_name} scale {target_scale}... [prefix_stream={encoded.stream_file_bytes} B | prompt payload={prompt_bits} bits | format=zero_header_v3]"
                            )
                            (scale_metric_one, decode_timings) = decompress_modified_from_file(
                                args,
                                infinity,
                                vae,
                                scale_schedule,
                                text_tokenizer,
                                text_encoder,
                                args.rec_path,
                                img_path,
                                img_name,
                                prefix_stream_path,
                                device,
                                h,
                                w,
                                lpips_evaluator,
                                dists_evaluator,
                                expected_prompt=text_for_decode,
                                expected_gt=gt_ms_idx_Bl if args.verify_stream else None,
                                max_scale=target_scale,
                                only_scale=target_scale)
                            scale_metrics.update(scale_metric_one)
                            values = scale_metric_one[str(target_scale)]
                            codec_timing_rows.append({
                                'image': img_name,
                                'scale': int(target_scale),
                                'codec_timing_mode': args.codec_timing_mode,
                                'stream_path': prefix_stream_path,
                                'stream_format': 'zero_header_v3',
                                'prompt_transport': 'standalone_zlib',
                                'enable_csa': int(args.enable_csa),
                                'csa_logit_fusion': int(args.csa_logit_fusion),
                                'csa_profile': csa_profile_json,
                                'rng_mode': values['rng_mode'],
                                'hash_seed_salt': values['hash_seed_salt'],
                                'decode_seed': values['decode_seed'],
                                'stream_prefix_bytes': values['stream_prefix_bytes'],
                                'stream_bpp': values['stream_bpp'],
                                'coding_payload_bpp': values['coding_payload_bpp'],
                                'image_payload_stream_bpp': values['image_payload_stream_bpp'],
                                'standalone_equivalent_bpp': values['standalone_equivalent_bpp'],
                                **encode_timings,
                                **decode_timings, 'reconstruct_s': values['reconstruct_s'],
                                'png_write_s': values['png_write_s'],
                                'metric_s': values['metric_s']
                            })
                    else:
                        (encoded, gt_ms_idx_Bl, encode_timings) = compress_to_file(
                            args, vae, scale_schedule, scale_q, infinity, text_tokenizer,
                            text_encoder, img_path, text, stream_path, text_for_decode, h, w)
                        print(
                            f"Processing {img_name}... [stream={encoded.stream_file_bytes} B | prompt payload={prompt_bits} bits | format=zero_header_v3]"
                        )
                        (scale_metrics, decode_timings) = decompress_modified_from_file(
                            args,
                            infinity,
                            vae,
                            scale_schedule,
                            text_tokenizer,
                            text_encoder,
                            args.rec_path,
                            img_path,
                            img_name,
                            stream_path,
                            device,
                            h,
                            w,
                            lpips_evaluator,
                            dists_evaluator,
                            expected_prompt=text_for_decode,
                            expected_gt=gt_ms_idx_Bl if args.verify_stream else None)
                        for (scale_idx, values) in scale_metrics.items():
                            codec_timing_rows.append({
                                'image': img_name,
                                'scale': int(scale_idx),
                                'codec_timing_mode': args.codec_timing_mode,
                                'stream_path': stream_path,
                                'stream_format': 'zero_header_v3',
                                'prompt_transport': 'standalone_zlib',
                                'enable_csa': int(args.enable_csa),
                                'csa_logit_fusion': int(args.csa_logit_fusion),
                                'csa_profile': csa_profile_json,
                                'rng_mode': values['rng_mode'],
                                'hash_seed_salt': values['hash_seed_salt'],
                                'decode_seed': values['decode_seed'],
                                'stream_prefix_bytes': values['stream_prefix_bytes'],
                                'stream_bpp': values['stream_bpp'],
                                'coding_payload_bpp': values['coding_payload_bpp'],
                                'image_payload_stream_bpp': values['image_payload_stream_bpp'],
                                'standalone_equivalent_bpp': values['standalone_equivalent_bpp'],
                                **encode_timings,
                                **decode_timings, 'reconstruct_s': values['reconstruct_s'],
                                'png_write_s': values['png_write_s'],
                                'metric_s': values['metric_s']
                            })
            dataset_metrics_data.append({
                'image_name': img_name,
                'text': text,
                'prompt_bpp': prompt_bpp,
                'stream_format': 'zero_header_v3',
                'prompt_transport': 'standalone_zlib',
                'codec_timing_mode': args.codec_timing_mode,
                'pure_gen_only': int(getattr(args, 'pure_gen_only', 0)),
                'csa_logit_fusion': int(args.csa_logit_fusion),
                'rng_mode': args.rng_mode,
                'hash_seed_salt': args.hash_seed_salt if args.rng_mode == 'prefix_hash' else '',
                'decode_seed': int(args.current_decode_seed)
                if args.rng_mode == 'stream_seed' else None,
                'scales_data': scale_metrics
            })
            bpp_list = [f"{v['bpp']:.8f}" for (k, v) in scale_metrics.items()]
            print(f'Finished {img_name}, Scales total bpp: {bpp_list}')
        scale_aggregates = {}
        for item in dataset_metrics_data:
            for (scale_idx, metrics) in item['scales_data'].items():
                if scale_idx not in scale_aggregates:
                    scale_aggregates[scale_idx] = {
                        'bpp': [],
                        'stream_bpp': [],
                        'coding_payload_bpp': [],
                        'image_payload_stream_bpp': [],
                        'standalone_equivalent_bpp': [],
                        'psnr': [],
                        'msssim': [],
                        'lpips': [],
                        'dists': []
                    }
                scale_aggregates[scale_idx]['bpp'].append(metrics['bpp'])
                if 'stream_bpp' in metrics:
                    scale_aggregates[scale_idx]['stream_bpp'].append(metrics['stream_bpp'])
                if 'coding_payload_bpp' in metrics:
                    scale_aggregates[scale_idx]['coding_payload_bpp'].append(
                        metrics['coding_payload_bpp'])
                for bpp_name in ('image_payload_stream_bpp', 'standalone_equivalent_bpp'):
                    if bpp_name in metrics:
                        scale_aggregates[scale_idx][bpp_name].append(metrics[bpp_name])
                scale_aggregates[scale_idx]['psnr'].append(metrics['psnr'])
                scale_aggregates[scale_idx]['msssim'].append(metrics['msssim'])
                scale_aggregates[scale_idx]['lpips'].append(metrics['lpips'])
                scale_aggregates[scale_idx]['dists'].append(metrics['dists'])
        average_metrics_summary = {}
        print(f'\n[*] 模型 Iter: {current_iter} 测试集平均指标汇总:')
        for (scale_idx, metric_lists) in scale_aggregates.items():
            avg_bpp = float(np.mean(metric_lists['bpp']))
            avg_psnr = float(np.mean(metric_lists['psnr']))
            avg_msssim = float(np.mean(metric_lists['msssim']))
            avg_lpips = float(np.mean(metric_lists['lpips']))
            avg_dists = float(np.mean(metric_lists['dists']))
            avg_stream_bpp = float(np.mean(
                metric_lists['stream_bpp'])) if metric_lists['stream_bpp'] else float('nan')
            avg_coding_payload_bpp = float(np.mean(metric_lists['coding_payload_bpp'])
                                           ) if metric_lists['coding_payload_bpp'] else float('nan')
            avg_image_payload_stream_bpp = float(
                np.mean(metric_lists['image_payload_stream_bpp']
                        )) if metric_lists['image_payload_stream_bpp'] else float('nan')
            avg_standalone_equivalent_bpp = float(
                np.mean(metric_lists['standalone_equivalent_bpp']
                        )) if metric_lists['standalone_equivalent_bpp'] else float('nan')
            average_metrics_summary[scale_idx] = {
                'avg_bpp': avg_bpp,
                'avg_stream_bpp': avg_stream_bpp,
                'avg_coding_payload_bpp': avg_coding_payload_bpp,
                'avg_image_payload_stream_bpp': avg_image_payload_stream_bpp,
                'avg_standalone_equivalent_bpp': avg_standalone_equivalent_bpp,
                'avg_psnr': avg_psnr,
                'avg_msssim': avg_msssim,
                'avg_lpips': avg_lpips,
                'avg_dists': avg_dists,
                'image_count': len(metric_lists['bpp'])
            }
            print(
                f'    Scale {scale_idx} | stream BPP: {avg_bpp:.4f} | PSNR: {avg_psnr:.4f} | MS-SSIM: {avg_msssim:.4f} | LPIPS: {avg_lpips:.4f} | DISTS: {avg_dists:.4f}'
            )
        final_json_output = {
            'model_iter': current_iter,
            'rng_mode': args.rng_mode,
            'stream_format': 'zero_header_v3',
            'prompt_transport': 'standalone_zlib',
            'codec_timing_mode': args.codec_timing_mode,
            'pure_gen_only': int(getattr(args, 'pure_gen_only', 0)),
            'model_path': args.model_path,
            'csa_config': csa_config,
            'hash_seed_salt': args.hash_seed_salt if args.rng_mode == 'prefix_hash' else '',
            'decode_seed_base': int(args.decode_seed_base),
            'summary': average_metrics_summary,
            'details': dataset_metrics_data
        }
        json_save_path = os.path.join(args.rec_path, f'metrics_iter_{current_iter}.json')
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(final_json_output, f, indent=4, ensure_ascii=False)
        df_summary = pd.DataFrame(average_metrics_summary).T
        df_summary.index.name = 'scale'
        avg_text_bpp = float(np.mean([item['prompt_bpp'] for item in dataset_metrics_data
                                      ])) if dataset_metrics_data else float('nan')
        df_summary['avg_text_bpp'] = avg_text_bpp
        df_summary['enable_csa'] = int(args.enable_csa)
        df_summary['csa_logit_fusion'] = int(args.csa_logit_fusion)
        df_summary['csa_profile'] = csa_profile_json
        df_summary['rng_mode'] = args.rng_mode
        df_summary['codec_timing_mode'] = args.codec_timing_mode
        df_summary['pure_gen_only'] = int(getattr(args, 'pure_gen_only', 0))
        df_summary['hash_seed_salt'] = args.hash_seed_salt if args.rng_mode == 'prefix_hash' else ''
        csv_save_path = os.path.join(args.rec_path, f'avg_metrics_iter_{current_iter}.csv')
        df_summary.to_csv(csv_save_path)
        if codec_timing_rows:
            timing_path = os.path.join(args.rec_path, f'codec_timing_iter_{current_iter}.csv')
            timing_summary_path = os.path.join(args.rec_path,
                                               f'codec_timing_iter_{current_iter}.summary.csv')
            timing_df = pd.DataFrame(codec_timing_rows)
            timing_df.to_csv(timing_path, index=False)
            numeric_columns = [
                column for column in timing_df.select_dtypes(include=[np.number]).columns.tolist()
                if column not in ('scale', 'decode_seed')
            ]
            timing_group_columns = ['codec_timing_mode', 'scale'
                                    ] if 'codec_timing_mode' in timing_df.columns else ['scale']
            timing_summary = timing_df.groupby(timing_group_columns,
                                               as_index=False)[numeric_columns].mean()
            timing_summary.to_csv(timing_summary_path, index=False)
            print(f'[*] Codec timing 已保存至: {timing_path}')
            print(f'[*] Codec timing 汇总已保存至: {timing_summary_path}')
        print(f'[*] 当前模型详细指标及平均值已保存至: {json_save_path}')
        print(f'[*] 平均值 CSV 已单独保存至: {csv_save_path}\n')
    print('\n[*] 所有可用模型已测试完毕。退出测试循环。')
    if args.keep_gpu_busy and json_data:
        _keep_gpu_busy(args, infinity, vae, text_tokenizer, text_encoder, json_data[0])


if __name__ == "__main__":
    main()
