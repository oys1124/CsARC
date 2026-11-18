"""Model initialization, checkpoint export and training-state restoration."""
from collections import OrderedDict
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as tdist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
    FullStateDictConfig, FullOptimStateDictConfig, StateDictType)


def read_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    except RuntimeError as exc:
        if 'mmap' not in str(exc):
            raise
        return torch.load(path, map_location='cpu', weights_only=False)


def extract_model_state(checkpoint):
    if 'trainer' in checkpoint:
        trainer = checkpoint['trainer']
        for key in ('gpt_fsdp', 'gpt_wo_ddp', 'gpt'):
            if key in trainer and trainer[key] is not None:
                return trainer[key]
        raise ValueError('No recognized model state in the full checkpoint')
    for key in ('gpt', 'model', 'state_dict'):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def convert_state_names(state, backbone_only=False):
    """Select checkpoint parameters and optionally keep backbone weights."""
    result = OrderedDict()
    ignored = []
    for name, value in state.items():
        if backbone_only and name.startswith(('csa_head.', 'csa_target_projector.')):
            continue
        if name in result:
            raise ValueError(f'Duplicate checkpoint parameter: {name}')
        result[name] = value
    return result, ignored


def initialize_model(model, checkpoint, allow_missing_csa=False):
    state, ignored = convert_state_names(extract_model_state(checkpoint),
                                        backbone_only=not model.enable_csa)
    target = model.state_dict()
    mismatch = [(k, tuple(v.shape), tuple(target[k].shape))
                for k, v in state.items() if k in target and v.shape != target[k].shape]
    if mismatch:
        raise ValueError(f'Checkpoint dimensions do not match the configuration: {mismatch}')
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys
               if not (allow_missing_csa and k.startswith(('csa_head.', 'csa_target_projector.')))]
    if missing or result.unexpected_keys:
        raise ValueError(f'Incomplete checkpoint: missing={missing}, unexpected={result.unexpected_keys}')
    return {'missing_auxiliary': list(result.missing_keys), 'ignored_legacy': ignored}


def rng_state():
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'])


def save_training_checkpoint(path, model, optimizer, config, step, epoch, next_batch,
                             legacy_iteration=None):
    distributed = tdist.is_initialized()
    rank = tdist.get_rank() if distributed else 0
    local_rng = rng_state()
    all_rng = [None] * tdist.get_world_size() if distributed else [local_rng]
    if distributed:
        tdist.all_gather_object(all_rng, local_rng)
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state = model.state_dict()
            optim = FSDP.optim_state_dict(model, optimizer)
    else:
        state = (model.module if hasattr(model, 'module') else model).state_dict()
        optim = optimizer.state_dict()
    if rank == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save({'format': 'csa-training-v1', 'model': state, 'optimizer': optim,
                    'config': config, 'step': step, 'epoch': epoch, 'next_batch': next_batch,
                    'legacy_iteration': legacy_iteration, 'rng_by_rank': all_rng}, temporary)
        temporary.replace(path)
    if distributed:
        tdist.barrier()
