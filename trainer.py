"""CSA training stages, tokenizer inputs and gradient control."""
from contextlib import nullcontext

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from infinity.models.csa import compute_csa_loss
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w


def configure_stage(model, stage, fsdp=False):
    """Configure stage-specific parameter groups before distributed wrapping.

    FSDP selects gradients through the stage loss. DDP and single-device
    training freeze inactive parameter groups before wrapping.
    """
    if stage not in {'target_ae', 'align'}:
        raise ValueError(f'Unknown CSA stage: {stage}')
    if fsdp:
        return
    for name, parameter in model.named_parameters():
        target = name.startswith('csa_target_projector.')
        parameter.requires_grad_(target if stage == 'target_ae' else not target)


class CSATrainer:
    def __init__(self, model, vae, args):
        self.model, self.vae, self.args = model, vae, args
        self.self_correction = BitwiseSelfCorrection(vae, args)

    def loss(self, images, text_condition, step):
        args = self.args
        # 1024-square images use 13 scales with tokenizer stride 16.
        schedule = [(1, h, w) for _, h, w in dynamic_resolution_h_w[1.0]['1M']['scales']]
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            raw, _, _ = self.vae.encode_for_raw_features(images, scale_schedule=schedule)
            inputs, bits, codes = self.self_correction.flip_requant(schedule, raw)
        target_codes = torch.cat(codes, dim=1).contiguous()
        target_bits = torch.cat(bits, dim=1).contiguous().long()
        amp = (torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=args.sharding != 'fsdp')
               if images.device.type == 'cuda' and args.bf16 else nullcontext())
        with amp:
            logits, aux = self.model(text_condition, inputs, scale_schedule=schedule,
                                     return_csa=True, csa_target_BLC=target_codes,
                                     csa_target_ae_only=args.stage == 'target_ae')
            return compute_csa_loss(
                logits, aux, target_codes, target_bits, schedule, args.stage, step,
                args.csa_weight, args.csa_scale_min, args.csa_scale_max,
                args.csa_align_warmup_steps,
            )


def clip_gradients(model, limit):
    if isinstance(model, FSDP):
        return model.clip_grad_norm_(limit)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), limit)
