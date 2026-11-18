"""Load the BSQ tokenizer, T5 text encoder and Infinity inference backbone."""
import os
from contextlib import nullcontext
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from infinity.models.infinity import Infinity
from infinity.models.basic import CrossAttnBlock, SelfAttnBlock
from infinity.models.sampling import sample_with_top_k_top_p_also_inplace_modifying_logits_
from infinity.models.bsq_vae.vae import vae_model
from infinity.utils.checkpoint import initialize_model, read_checkpoint
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


def _device_from(args=None):
    value = getattr(args, 'device', None) if args is not None else None
    return torch.device(value or ('cuda' if torch.cuda.is_available() else 'cpu'))


def load_tokenizer(t5_path='', device=None):
    from transformers import AutoTokenizer, T5EncoderModel
    if not t5_path:
        raise ValueError('Provide a local --text_encoder_ckpt directory')
    device = torch.device(device or _device_from())
    tokenizer = AutoTokenizer.from_pretrained(t5_path, legacy=True, local_files_only=True)
    tokenizer.model_max_length = 512
    encoder = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float16,
                                            local_files_only=True).to(device)
    encoder.eval().requires_grad_(False)
    return tokenizer, encoder


def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False):
    if enable_positive_prompt:
        prompt = aug_with_positive_prompt(prompt)
    device = next(text_encoder.parameters()).device
    tokens = text_tokenizer(text=[prompt], max_length=512, padding='max_length',
                            truncation=True, return_tensors='pt')
    input_ids = tokens.input_ids.to(device, non_blocking=True)
    mask = tokens.attention_mask.to(device, non_blocking=True)
    with torch.no_grad():
        features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
    lengths = mask.sum(-1).tolist()
    cumulative = F.pad(mask.sum(-1).to(torch.int32).cumsum_(0), (1, 0))
    compact = torch.cat([feature[:length] for length, feature in zip(lengths, features.unbind(0))])
    return compact, lengths, cumulative, max(lengths)


def aug_with_positive_prompt(prompt):
    for key in ['man', 'woman', 'men', 'women', 'boy', 'girl', 'child', 'person', 'human', 'adult', 'teenager', 'employee', 
                'employer', 'worker', 'mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather', 'son', 'daughter']:
        if key in prompt:
            prompt = prompt + '. very smooth faces, good looking faces, face to the camera, perfect facial features'
            break
    return prompt


def load_visual_tokenizer(args):
    device = _device_from(args)
    if int(args.vae_type) != 32 or int(getattr(args, 'apply_spatial_patchify', 0)) != 0:
        raise ValueError('Expected a d=32 BSQ tokenizer with apply_spatial_patchify=0')
    checkpoint = read_checkpoint(args.vae_path)
    vae = vae_model(checkpoint, 'dynamic', 32, 2**32, test_mode=True,
                    patch_size=16, encoder_ch_mult=[1, 2, 4, 4, 4],
                    decoder_ch_mult=[1, 2, 4, 4, 4]).to(device)
    return vae


def load_transformer(vae, args):
    if getattr(args, 'model_type', 'infinity_2b') != 'infinity_2b':
        raise ValueError('Expected model_type=infinity_2b')
    if getattr(args, 'checkpoint_type', 'torch') != 'torch':
        raise ValueError('Provide a full/slim/exported PyTorch checkpoint')
    device = _device_from(args)
    # Use fused norms on CUDA and unfused norms on CPU.
    fused_norm = bool(getattr(args, 'fused_norm', True)) and device.type == 'cuda'
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with torch.no_grad(), amp:
        model = Infinity(vae_local=vae, text_channels=int(getattr(args, 'text_channels', 2048)),
            text_maxlen=512, embed_dim=2048, depth=32, num_heads=16, mlp_ratio=4,
            drop_path_rate=.1, block_chunks=8, shared_aln=True,
            raw_scale_schedule=None, checkpointing='full-block',
            customized_flash_attn=False, fused_norm=fused_norm, pad_to_multiplier=128,
            use_flex_attn=bool(getattr(args, 'use_flex_attn', False)),
            add_lvl_embeding_only_first_block=int(getattr(args, 'add_lvl_embeding_only_first_block', 1)),
            use_bit_label=1,
            rope2d_each_sa_layer=int(getattr(args, 'rope2d_each_sa_layer', 1)),
            rope2d_normalized_by_hw=int(getattr(args, 'rope2d_normalized_by_hw', 2)),
            pn=getattr(args, 'pn', '1M'), apply_spatial_patchify=0,
            inference_mode=True, train_h_div_w_list=[1.0], enable_csa=False).to(device)
        # Cast Transformer blocks to BF16 when enabled.
        if bool(getattr(args, 'bf16', False)):
            for block in model.unregistered_blocks:
                block.bfloat16()
        report = initialize_model(model, read_checkpoint(args.model_path))
        model.eval().requires_grad_(False)
        model.rng = torch.Generator(device=device)
    print('CSA inference backbone loaded:', report)
    return model
