# Code-State Alignment (CSA)

Code-State Alignment for Visual Autoregressive Image Compression.

CSA uses an Infinity-2B backbone, a frozen d=32 BSQ tokenizer and a 256-dimensional alignment space.

## Contents

- [Environment Setup](#environment-setup)
- [Pretrained Models](#pretrained-models)
- [Training](#training)
- [Inference](#inference)
- [Acknowledgement](#acknowledgement)

## Environment Setup

Use Python 3.10 and a CUDA-enabled PyTorch installation. FlashAttention must match the installed PyTorch and CUDA versions.

```bash
conda create -n csa python=3.10
conda activate csa

pip install torch==2.8.0 torchvision==0.23.0
pip install -r requirements-inference.txt
```

Run all commands below from the project root. `requirements-inference.txt` includes the training dependencies.

## Pretrained Models

Place the downloaded weights under `ckpt/`.

| Model | Usage | Local path | Download |
|---|---|---|---|
| CSA (D=256) | Image compression and reconstruction | `ckpt/csa.pth` | **To be added** |
| Infinity-2B | Training initialization | `ckpt/infinity_2b_reg.pth` | **To be added** |
| BSQ tokenizer (d=32) | Training and inference | `ckpt/infinity_vae_d32reg.pth` | **To be added** |
| Flan-T5-XL | Caption encoding | `ckpt/flan-t5-xl/` | **To be added** |

The Flan-T5-XL directory must contain both model weights and tokenizer files.

## Training

Training consists of target-code autoencoder construction followed by hidden-state alignment. Both stages use a projection dimension of 256.

### Data Preparation

Prepare training images and precomputed T5 text embeddings with width 2048. The embedding directory must have the following structure:

```text
DATA_ROOT/
├── samples.jsonl
├── samples_offsets.pt
└── shards/
    ├── shard_000000.meta.pt
    └── <kv_bin from the shard metadata>
```

Each `samples.jsonl` record contains an image path and embedding index:

```json
{"path": "/path/to/image.jpg", "shard": 0, "idx": 0}
```

`samples_offsets.pt` stores int64 byte offsets into the JSONL file. Each shard metadata file contains `rows`, `hidden_size`, `dtype`, `kv_bin`, `offsets` and `lens`. Image preprocessing resizes and center-crops inputs to 1024×1024.

Set the training paths:

```bash
DATA_ROOT=/path/to/embedding_mmap
VAE_CKPT=ckpt/infinity_vae_d32reg.pth
PRETRAINED=ckpt/infinity_2b_reg.pth
```

### Stage 1: Target-Code Autoencoder

Train the code encoder and decoder:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_csa.sh \
  --config configs/paper_proj256_target_ae.json \
  --proc_data_path "$DATA_ROOT" \
  --vae_ckpt "$VAE_CKPT" \
  --init_checkpoint "$PRETRAINED" \
  --output_dir outputs/csa_proj256_target_ae
```

### Stage 2: Code-State Alignment

Initialize from Stage 1 and train the backbone and hidden-state projector with the target-code autoencoder fixed:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_csa.sh \
  --config configs/paper_proj256_align.json \
  --proc_data_path "$DATA_ROOT" \
  --vae_ckpt "$VAE_CKPT" \
  --init_checkpoint outputs/csa_proj256_target_ae/last.pth \
  --output_dir outputs/csa_proj256_align
```

The JSON configurations set a per-GPU batch size of 10, learning rate of `6e-5`, and 2000 updates per stage. CLI arguments override the configuration values. Set `CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE` to use multiple GPUs.

Resume an interrupted alignment run:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_csa.sh \
  --config configs/paper_proj256_align.json \
  --proc_data_path "$DATA_ROOT" \
  --vae_ckpt "$VAE_CKPT" \
  --resume outputs/csa_proj256_align/last.pth \
  --output_dir outputs/csa_proj256_align
```

`--resume` requires a full training checkpoint and the same training configuration and world size. Use `--init_checkpoint` for weights-only initialization.

Inspect the configuration without loading data or models:

```bash
python train.py --config configs/paper_proj256_align.json --dry_run
```

## Inference

### Inputs

Prepare 1024×1024 images and a caption JSON file:

```json
[
  {"img_path": "/path/to/image.png", "txt": "a clear photo of a garden"}
]
```

Set the inference paths:

```bash
MODEL_CKPT=ckpt/csa.pth
VAE_CKPT=ckpt/infinity_vae_d32reg.pth
T5_PATH=ckpt/flan-t5-xl
CAPTION_JSON=/path/to/captions.json
```

### Scale-Wise Transmission

```bash
CUDA_VISIBLE_DEVICES=0 python eval_scale_stream.py \
  --model_path "$MODEL_CKPT" \
  --vae_path "$VAE_CKPT" \
  --text_encoder_ckpt "$T5_PATH" \
  --caption_json "$CAPTION_JSON" \
  --out_dir outputs/csa_scale_eval \
  --rng_mode prefix_hash \
  --tau 1.0 \
  --decode_cfg 3.0 --decode_tau 0.5 \
  --verify_stream 1
```

This command writes `.iraw` files, reads them back for entropy decoding, and reconstructs images from all 13 scale prefixes. Captions are included in the stream and counted in bpp. Outputs include reconstructed images, per-image metrics and aggregate CSV/JSON reports.

### HPRC Rate Control

```bash
CUDA_VISIBLE_DEVICES=0 python eval_hprc.py \
  --model_path "$MODEL_CKPT" \
  --vae_path "$VAE_CKPT" \
  --text_encoder_ckpt "$T5_PATH" \
  --caption_json "$CAPTION_JSON" \
  --out_dir outputs/csa_hprc \
  --target_bpps 0.0008,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.15,0.2 \
  --ms_list entropy --bit_select_order high \
  --seed_policy prefix_hash \
  --decode_cfg 3.0 --decode_tau 0.5 --prob_tau 0.5 \
  --write_streams 1 --save_recon 1
```

HPRC writes and reads `.rcp` files for each image/rate pair. Outputs include streams, reconstructions, `rate_control_progressive.csv` and an aggregate summary. Add `--resume 1` with the same configuration to continue an interrupted evaluation.

For a quick run, add `--max_images 1` to either inference command. Additional parameters and output definitions are described in [INFERENCE.md](INFERENCE.md).

### Export Inference Weights

```bash
python tools/convert_checkpoint.py \
  --input outputs/csa_proj256_align/last.pth \
  --output ckpt/csa.pth \
  --backbone_only
```

Use the exported file as `--model_path` in either inference command.

## Acknowledgement

This project uses the Infinity backbone and BSQ tokenizer. See [LICENSE](LICENSE) for licensing information.
