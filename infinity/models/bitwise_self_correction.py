"""Random bit perturbation and requantization for CSA training.

Targets are quantized before corruption. Later targets are requantized against
the accumulated corrupted prefix.
"""
import numpy as np
import torch
import torch.nn.functional as F


class BitwiseSelfCorrection:
    def __init__(self, vae, args):
        self.vae = vae
        self.noise_apply_layers = args.noise_apply_layers
        self.noise_apply_strength = args.noise_apply_strength
        self.noise_apply_requant = args.noise_apply_requant

    @torch.no_grad()
    def flip_requant(self, scale_schedule, raw_features):
        with torch.amp.autocast('cuda', enabled=False):
            batch = raw_features.shape[0]
            codes_out = raw_features.unsqueeze(2) if raw_features.dim() == 4 else raw_features
            cumulative = 0
            inputs, labels, targets = [], [], []
            for si, size in enumerate(scale_schedule):
                residual = codes_out - cumulative
                if si != len(scale_schedule) - 1:
                    residual = F.interpolate(residual, size=size,
                                             mode=self.vae.quantizer.z_interplote_down).contiguous()
                quantized, _, bits, _ = self.vae.quantizer.lfq(residual)
                target = quantized.detach()
                targets.append(target.reshape(*target.shape[:2], -1).permute(0, 2, 1))
                labels.append(bits.reshape(batch, -1, self.vae.codebook_dim))
                if si < self.noise_apply_layers:
                    # Draw the corruption probability on a 0.01-spaced uniform grid.
                    strength = np.random.randint(0, int(round(100 * self.noise_apply_strength)) + 1) * .01
                    mask = torch.rand(*bits.shape, device=bits.device) < strength
                    corrupted_bits = bits.clone()
                    corrupted_bits[mask] = 1 - corrupted_bits[mask]
                    if self.noise_apply_requant:
                        quantized = self.vae.quantizer.lfq.indices_to_codes(
                            corrupted_bits, label_type='bit_label')
                cumulative = cumulative + F.interpolate(
                    quantized, size=scale_schedule[-1],
                    mode=self.vae.quantizer.z_interplote_up).contiguous()
                if si < len(scale_schedule) - 1:
                    next_input = F.interpolate(cumulative, size=scale_schedule[si + 1],
                                              mode=self.vae.quantizer.z_interplote_up).contiguous()
                    inputs.append(next_input.reshape(*next_input.shape[:2], -1).permute(0, 2, 1))
            return torch.cat(inputs, dim=1), labels, targets
