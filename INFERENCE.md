# Inference

## Setup and inputs

```bash
pip install -r requirements-inference.txt
```

Run commands from the project root. Prepare:

- `--model_path`: an Infinity-2B checkpoint, CSA training checkpoint or backbone export.
- `--vae_path`: frozen d32 BSQ tokenizer weights.
- `--text_encoder_ckpt`: a local Flan-T5-XL directory containing the model and tokenizer files.
- `--caption_json`: image paths and captions.

Caption files support JSON lists, JSONL records, or dictionaries mapping image paths to captions. For example:

```json
[
  {"img_path": "/path/to/image.png", "txt": "a clear photo of a garden"}
]
```

The keys `image_path` / `path` and `caption` / `text` are also accepted. To use one caption for a directory, pass `--data_dir /path/to/images --fallback_prompt "a clear photo"` instead of `--caption_json`.

Use 1024×1024 square images for scale-wise evaluation. The model uses 13 scales, d=32 bit labels and no spatial patchification. Evaluation metrics require the LPIPS and DISTS pretrained weights.

## Scale-wise evaluation

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_scale.sh \
  --model_path /path/to/checkpoint.pth \
  --vae_path /path/to/infinity_vae_d32reg.pth \
  --text_encoder_ckpt /path/to/flan-t5-xl \
  --caption_json /path/to/captions.json \
  --out_dir outputs/csa_scale_eval \
  --rng_mode prefix_hash \
  --tau 1.0 --decode_cfg 3.0 --decode_tau 0.5
```

The encoder writes `.iraw` files, and the decoder reads those files to reconstruct images from each transmitted scale prefix. Captions are normalized through T5 token IDs and stored in the stream using the shorter UTF-8/zlib representation. Stream bpp includes the caption and scale descriptors. Each scale uses RangeCoder or a raw-bit payload when arithmetic coding would increase its size.

| Option | Behavior |
|---|---|
| `--codec_timing_mode multirate` | Encode all 13 scales once and evaluate all prefixes; default. |
| `--codec_timing_mode prefix_only` | Encode and decode each prefix separately. |
| `--verify_stream 1` | Compare recovered code bits with encoder ground truth. |
| `--rng_mode prefix_hash` | Derive sampling seeds from the caption and transmitted prefix; default. |
| `--rng_mode stream_seed` | Use fixed sampling seed 0. |
| `--tau` | Entropy-coding temperature; default 1.0. |
| `--decode_cfg`, `--decode_tau` | Reconstruction guidance and temperature; defaults 3.0 and 0.5. |
| `--max_images N` | Evaluate at most N images; 0 evaluates all. |
| `--save_recon 0` | Compute metrics without saving reconstructed PNGs. |
| `--sweep_checkpoints` | Evaluate sibling checkpoints whose `-iterN-` filename index is at least the selected checkpoint's index. |
| `--pure_gen_only 1` | Evaluate generation without transmitted image scales. |

Streams, reconstruction images, per-image metrics, aggregate metrics and timing CSVs are written beneath `--out_dir`. `stream_bpp` counts the file prefix, `coding_payload_bpp` counts the encoded image payload, and `image_payload_stream_bpp` includes image-scale descriptors. `standalone_equivalent_bpp` equals `stream_bpp`.

Check inputs and CLI configuration without loading models:

```bash
python eval_scale_stream.py \
  --model_path /path/to/checkpoint.pth \
  --vae_path /path/to/infinity_vae_d32reg.pth \
  --text_encoder_ckpt /path/to/flan-t5-xl \
  --caption_json /path/to/captions.json --dry_run
```

## HPRC rate control

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_hprc.sh \
  --model_path /path/to/checkpoint.pth \
  --vae_path /path/to/infinity_vae_d32reg.pth \
  --text_encoder_ckpt /path/to/flan-t5-xl \
  --caption_json /path/to/captions.json \
  --out_dir outputs/csa_hprc \
  --target_bpps 0.0008,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.15,0.2 \
  --ms_list entropy --bit_select_order high \
  --seed_policy prefix_hash \
  --decode_cfg 3.0 --decode_tau 0.5 --prob_tau 0.5
```

HPRC allocates bits under the target-rate budget and writes `.rcp` files containing the caption, a 4-byte target-payload budget and the RangeCoder payload. Decoding first reproduces the CFG-free entropy model to recover transmitted bits, then reconstructs with the selected CFG and temperature while forcing those bits into their decoded positions.

| Option | Default | Purpose |
|---|---|---|
| `--target_bpps` | Comma-separated rate list | Target bits per pixel. |
| `--decode_cfg`, `--decode_tau` | 3.0, 0.5 | Reconstruction guidance and temperature. |
| `--prob_tau` | 0.5 | Entropy-model temperature. |
| `--budget_rate_scale` | 1.6 | Rate-cost multiplier for bit allocation. |
| `--rate_overhead_bits` | 256 | Reserved bits deducted from the payload budget. |
| `--seed_policy` | `prefix_hash` | Deterministic completion seed policy. |
| `--write_streams` | 1 | Write and read `.rcp` files during evaluation. |
| `--save_recon` | 1 | Save reconstructed images. |
| `--stream_target_payload_header` | 1 | Include the 4-byte target-payload budget. |
| `--prompt_bits_mode` | `zlib` | Caption normalization and model-bit accounting; `arith` requires `torchac`. |

Rate-cost and overhead schedules use comma-separated `target_lower_bound:value` pairs, for example:

```bash
--budget_rate_scale_schedule '0.0:1.6,0.15:1.65' \
--rate_overhead_schedule '0.0:256,0.15:4096'
```

Outputs include streams, reconstructions, per-image/rate CSV records, an aggregate summary and `rate_control_progressive.config.json`. Add `--resume 1` with the same configuration to process pending image/rate pairs. `--dry_run 1` reports pending pairs without loading models or writing output.

## RangeCoder acceleration

The Python RangeCoder runs without a compiled extension. Build the C++ extension for acceleration:

```bash
python codec/setup_arithmetic_fast.py build_ext --inplace
```

## Tests

```bash
python -m unittest discover -s tests -v
```
