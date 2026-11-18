#!/usr/bin/env python3
"""CSA target-construction and alignment training."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as tdist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn.parallel import DistributedDataParallel as DDP

from infinity.dataset.build_stage2_dataset import build_stage2_dataloader, move_text_cond_tuple_to_device
from infinity.models.infinity import Infinity, MultipleLayers
from infinity.models.init_param import init_weights
from infinity.models.bsq_vae.vae import vae_model
from infinity.utils import dist
from infinity.utils.checkpoint import (read_checkpoint, initialize_model, restore_rng,
                                       save_training_checkpoint)
from infinity.utils.lr_control import filter_params, lr_wd_annealing
from trainer import CSATrainer, configure_stage, clip_gradients


CONFIG_ROOT = Path(__file__).resolve().parent / 'configs'


def parse_args(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config', type=Path, default=CONFIG_ROOT / 'paper_proj256_align.json')
    selected, _ = preliminary.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    parser.add_argument('--stage', choices=['target_ae', 'align'], default='align')
    parser.add_argument('--proc_data_path', default='')
    parser.add_argument('--vae_ckpt', default='')
    parser.add_argument('--init_checkpoint', default='', help='Model weights for training initialization')
    parser.add_argument('--resume', default='', help='Explicit CSA training checkpoint; restores optimizer, position and RNG')
    parser.add_argument('--output_dir', type=Path, default=Path('outputs/csa_proj256_align'))
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--schedule_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size per GPU')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--prefetch_factor', type=int, default=16)
    parser.add_argument('--base_lr', type=float, default=6e-5)
    parser.add_argument('--lr_mode', choices=['legacy_scaled', 'absolute'], default='legacy_scaled')
    parser.add_argument('--weight_decay', type=float, default=.005)
    parser.add_argument('--adam_betas', type=float, nargs=2, default=[.9, .97])
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--sharding', choices=['fsdp', 'ddp', 'none'], default='fsdp')
    parser.add_argument('--bf16', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fused_optimizer', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_flex_attn', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--checkpointing', choices=['full-block', 'none'], default='full-block')
    parser.add_argument('--csa_projected_dim', type=int, default=256)
    parser.add_argument('--csa_head_hidden_dim', type=int, default=1024)
    parser.add_argument('--csa_target_projector_hidden_dim', type=int, default=1024)
    parser.add_argument('--csa_weight', type=float, default=.03)
    parser.add_argument('--csa_scale_min', type=int, default=0)
    parser.add_argument('--csa_scale_max', type=int, default=11)
    parser.add_argument('--csa_align_warmup_steps', type=int, default=1000)
    parser.add_argument('--noise_apply_strength', type=float, default=.3)
    parser.add_argument('--noise_apply_layers', type=int, default=13)
    parser.add_argument('--noise_apply_requant', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--save_before_step', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dry_run', action='store_true', help='Print and validate configuration without opening data or CUDA')
    config = json.loads(selected.config.read_text())
    known = {action.dest for action in parser._actions}
    unknown = set(config) - known
    if unknown:
        parser.error(f'Unknown configuration keys: {sorted(unknown)}')
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    args.output_dir = Path(args.output_dir)
    if not 0 <= args.csa_scale_min <= args.csa_scale_max <= 12:
        parser.error('CSA scale indices must satisfy 0 <= min <= max <= 12')
    if args.csa_projected_dim <= 0 or min(args.csa_head_hidden_dim, args.csa_target_projector_hidden_dim) <= 0:
        parser.error('CSA dimensions must be positive')
    if not 0 <= args.noise_apply_strength <= 1 or args.csa_weight < 0:
        parser.error('Invalid noise strength or CSA weight')
    if min(args.max_steps, args.batch_size, args.save_every, args.log_every, args.schedule_epochs) <= 0:
        parser.error('Steps, batch size and intervals must be positive')
    if args.csa_align_warmup_steps < 0 or args.workers < 0:
        parser.error('Warmup and worker count must be nonnegative')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    args.effective_lr = args.base_lr * args.batch_size * world_size / 256 if args.lr_mode == 'legacy_scaled' else args.base_lr
    args.config = str(args.config)
    if not args.dry_run:
        for name in ('proc_data_path', 'vae_ckpt'):
            if not getattr(args, name) or not Path(getattr(args, name)).exists():
                parser.error(f'Provide an existing --{name}')
        if not (args.init_checkpoint or args.resume):
            parser.error('Provide --init_checkpoint (or --resume for an existing CSA run)')
        for name in ('init_checkpoint', 'resume'):
            if getattr(args, name) and not Path(getattr(args, name)).is_file():
                parser.error(f'File does not exist: --{name}')
    return args


def config_dict(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def validate_resume_config(previous, args):
    # Exact resume requires matching random bit perturbation.
    if previous.get('flip_mode', 'random') != 'random':
        raise ValueError('Cannot resume a checkpoint with non-random bit self-correction; '
                         'use --init_checkpoint to start a new random-training run')
    for key in ('stage', 'csa_projected_dim', 'csa_head_hidden_dim',
                'csa_target_projector_hidden_dim', 'sharding', 'batch_size',
                'csa_scale_min', 'csa_scale_max', 'csa_align_warmup_steps', 'csa_weight',
                'base_lr', 'lr_mode', 'weight_decay', 'adam_betas', 'grad_clip',
                'bf16', 'use_flex_attn', 'checkpointing',
                'noise_apply_strength', 'noise_apply_layers', 'noise_apply_requant',
                'schedule_epochs'):
        if previous[key] != getattr(args, key):
            raise ValueError(f'Resume configuration mismatch for {key}: {previous[key]} vs {getattr(args, key)}')


def build_model(args, vae):
    # Infinity-2B: 32 blocks grouped into 8 FSDP chunks.
    model = Infinity(vae, text_channels=2048, text_maxlen=512,
                     embed_dim=2048, depth=32, num_heads=16, mlp_ratio=4,
                     drop_path_rate=.1, shared_aln=True, head_aln=True,
                     cond_drop_rate=.1, cos_attn=True, raw_scale_schedule=None,
                     customized_flash_attn=False, fused_mlp=False, fused_norm=False,
                     block_chunks=8, checkpointing=None if args.checkpointing == 'none' else args.checkpointing,
                     pad_to_multiplier=128, use_flex_attn=args.use_flex_attn,
                     batch_size=args.batch_size, add_lvl_embeding_only_first_block=1,
                     use_bit_label=1, rope2d_each_sa_layer=1, rope2d_normalized_by_hw=2,
                     pn='1M', train_h_div_w_list=[1.0], always_training_scales=100,
                     apply_spatial_patchify=0, enable_csa=True,
                     csa_projected_dim=args.csa_projected_dim,
                     csa_head_hidden_dim=args.csa_head_hidden_dim,
                     csa_target_projector_hidden_dim=args.csa_target_projector_hidden_dim).to(dist.get_device())
    init_weights(model, other_std=math.sqrt(1 / model.C / 3))
    model.special_init(aln_init=1e-3, aln_gamma_init=5e-6, scale_head=.02, scale_proj=1)
    return model


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Full Infinity-2B training requires CUDA; use --dry_run to inspect the recipe')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'last.pth').exists() and not args.resume:
        raise ValueError('Output already contains last.pth; use --resume explicitly or choose a new output_dir')
    dist.init_distributed_mode(str(args.output_dir))
    if args.sharding == 'fsdp' and not dist.initialized():
        raise ValueError('FSDP requires torchrun, including for a single GPU')
    if args.sharding == 'none' and dist.get_world_size() != 1:
        raise ValueError('Use fsdp or ddp when WORLD_SIZE > 1')
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.device = str(dist.get_device())
    bundle = build_stage2_dataloader('proc_memmap', args.proc_data_path, [1024],
                args.batch_size, args.workers, seed=0 if args.seed is None else args.seed,
                prefetch_factor=args.prefetch_factor, memmap_cache_size=2,
                distributed=dist.initialized(), drop_last=True)
    if len(bundle.dataloader) == 0:
        raise ValueError('Dataset contains fewer samples than one training batch')
    vae_weights = read_checkpoint(args.vae_ckpt)
    vae = vae_model(vae_weights, 'dynamic', 32, 2**32, test_mode=True,
                    patch_size=16, encoder_ch_mult=[1, 2, 4, 4, 4],
                    decoder_ch_mult=[1, 2, 4, 4, 4]).to(args.device)
    del vae_weights
    model = build_model(args, vae)
    checkpoint = read_checkpoint(args.resume or args.init_checkpoint)
    if args.resume:
        if checkpoint.get('format') != 'csa-training-v1':
            raise ValueError('Legacy/slim weights support --init_checkpoint; exact --resume requires a CSA training checkpoint')
        validate_resume_config(checkpoint['config'], args)
        if len(checkpoint['rng_by_rank']) != dist.get_world_size():
            raise ValueError('Exact resume requires the same world size')
    print('Weight loading:', initialize_model(model, checkpoint,
          allow_missing_csa=(args.stage == 'target_ae' and not args.resume)))
    configure_stage(model, args.stage, fsdp=args.sharding == 'fsdp')
    ndim = {name: parameter.ndim for name, parameter in model.named_parameters()}
    if args.sharding == 'fsdp':
        model = FSDP(model, device_id=dist.get_local_rank(),
                     sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                     auto_wrap_policy=ModuleWrapPolicy([MultipleLayers]),
                     use_orig_params=True, sync_module_states=True, limit_all_gathers=True)
    elif args.sharding == 'ddp' and dist.initialized():
        model = DDP(model, device_ids=[dist.get_local_rank()], broadcast_buffers=False)
    _, _, groups = filter_params(model, ndim, nowd_keys={
        'cfg_uncond', 'pos_start', 'lvl_embed', 'gamma', 'beta', 'ada_gss',
        'text_proj_for_sos.ca.mat_q'}, allow_no_grad=True)
    optimizer = torch.optim.AdamW(groups, lr=args.effective_lr,
                                 betas=tuple(args.adam_betas), fused=args.fused_optimizer)
    step, epoch, next_batch, pending_rng = 0, 0, 0, None
    if args.resume:
        optim = checkpoint['optimizer']
        if isinstance(model, FSDP):
            optim = FSDP.optim_state_dict_to_load(model, optimizer, optim)
        optimizer.load_state_dict(optim)
        step, epoch, next_batch = checkpoint['step'], checkpoint['epoch'], checkpoint['next_batch']
        pending_rng = checkpoint['rng_by_rank'][dist.get_rank()]
    del checkpoint
    model.train()
    trainer = CSATrainer(model, vae, args)
    optimizer.zero_grad(set_to_none=True)
    config = config_dict(args)
    if dist.is_master():
        (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    schedule_steps = max(2, args.schedule_epochs * len(bundle.dataloader))
    started = time.monotonic()
    while step < args.max_steps:
        if bundle.sampler is not None:
            bundle.sampler.set_epoch(epoch)
        iterator = iter(bundle.dataloader)
        for _ in range(next_batch):
            next(iterator)
        # Restore after constructing/skipping the iterator, which consumes CPU RNG.
        if pending_rng is not None:
            restore_rng(pending_rng)
            pending_rng = None
        for batch_index in range(next_batch, len(bundle.dataloader)):
            images, text = next(iterator)
            if args.save_before_step and (step + 1) % args.save_every == 0:
                save_training_checkpoint(args.output_dir / f'csa-iter{step+1:06d}.pth',
                    model, optimizer, config, step, epoch, batch_index, legacy_iteration=step+1)
            images = images.to(args.device, non_blocking=True)
            text = move_text_cond_tuple_to_device(text, torch.device(args.device))
            lr_wd_annealing('lin0', optimizer, args.effective_lr, args.weight_decay,
                           args.weight_decay, step, 1e-8 * len(bundle.dataloader),
                           max(schedule_steps, step + 2), wp0=.005, wpe=1.)
            loss, metrics = trainer.loss(images, text, step)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {step}')
            loss.backward()
            grad_norm = clip_gradients(model, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            next_batch = batch_index + 1
            if next_batch == len(bundle.dataloader):
                save_epoch, save_batch = epoch + 1, 0
            else:
                save_epoch, save_batch = epoch, next_batch
            if not args.save_before_step and step % args.save_every == 0:
                save_training_checkpoint(args.output_dir / f'csa-iter{step:06d}.pth',
                    model, optimizer, config, step, save_epoch, save_batch)
            if step == 1 or step % args.log_every == 0:
                values = {key: float(value) for key, value in metrics.items()}
                values.update(step=step, stage=args.stage, lr=args.effective_lr,
                              grad_norm=float(grad_norm), elapsed_s=time.monotonic() - started)
                if dist.initialized():
                    for key in ('loss', 'bit_ce', 'align', 'reconstruction'):
                        scalar = loss.new_tensor(values[key])
                        tdist.all_reduce(scalar)
                        values[key] = float(scalar) / dist.get_world_size()
                if dist.is_master():
                    with (args.output_dir / 'metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(values) + '\n')
                    print('CSA', values)
            if step >= args.max_steps:
                save_training_checkpoint(args.output_dir / 'last.pth', model, optimizer,
                                         config, step, save_epoch, save_batch)
                return
        epoch += 1
        next_batch = 0


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps(config_dict(args), indent=2))
        return
    try:
        run(args)
    finally:
        dist.finalize()


if __name__ == '__main__':
    main()
