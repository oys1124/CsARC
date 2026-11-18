"""Code-state alignment projectors, target autoencoder and training losses.

E_q and P_h use LayerNorm -> Linear -> GELU -> Linear.
D_q uses Linear -> GELU -> Linear.
"""
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CSAHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=1024, dropout=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        layers = [norm_layer(in_dim), nn.Linear(in_dim, hidden_dim),
                  nn.GELU(approximate='tanh')]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CSATargetAutoencoder(nn.Module):
    def __init__(self, in_dim=32, out_dim=256, hidden_dim=1024, dropout=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.encoder = CSAHead(in_dim, out_dim, hidden_dim, dropout, norm_layer)
        layers = [nn.Linear(out_dim, hidden_dim), nn.GELU(approximate='tanh')]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, in_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        z = F.normalize(self.encoder(x.float()), dim=-1)
        return {'target': z, 'recon': self.decoder(z)}

    def decode(self, z):
        return self.decoder(F.normalize(z.float(), dim=-1))


def token_weights(scale_schedule, device):
    """Compute sqrt(N_last / N_k) scale weights normalized over positions."""
    last_area = np.sqrt(int(np.prod(scale_schedule[-1])))
    values = []
    for pt, ph, pw in scale_schedule:
        count = pt * ph * pw
        values.extend([last_area / np.sqrt(count)] * count)
    weights = torch.tensor(values, device=device)[None, :]
    return weights / weights.sum()


def compute_csa_loss(logits, aux, target_codes, target_bits, scale_schedule,
                     stage, step, csa_weight=0.03, csa_scale_min=0,
                     csa_scale_max=11, csa_align_warmup_steps=1000):
    """Compute target-autoencoder or hidden-state alignment loss.

    Reconstruction loss sums squared error over code coordinates.
    Alignment uses detached, normalized target embeddings.
    """
    weights = token_weights(scale_schedule, target_codes.device)
    mask = torch.cat([
        torch.full((pt * ph * pw,), csa_scale_min <= k <= csa_scale_max,
                   device=target_codes.device, dtype=torch.bool)
        for k, (pt, ph, pw) in enumerate(scale_schedule)
    ])[None, :]
    # Form masked weights and their denominator in the prediction dtype.
    prediction = aux.get('csa_pred', aux['csa_projected_target'])
    csa_weights = weights.to(prediction.dtype) * mask.to(prediction.dtype)
    denominator = csa_weights.sum().clamp_min(1e-12).float()

    def reduce(values):
        return (values * csa_weights.float()).sum(-1).div(denominator).mean()

    zero = target_codes.new_tensor(0.0)
    if stage == 'target_ae':
        reconstruction = F.mse_loss(aux['csa_target_recon'].float(),
                                   target_codes.float(), reduction='none')
        reconstruction = reconstruction.mean(-1) * target_codes.shape[-1]
        reconstruction = reduce(reconstruction)
        # Target construction minimizes unscaled reconstruction loss.
        return reconstruction, {'loss': reconstruction.detach(), 'bit_ce': zero,
                                'align': zero, 'reconstruction': reconstruction.detach(),
                                'align_factor': 0.0}
    if stage != 'align':
        raise ValueError(f'Unsupported CSA stage: {stage}')
    batch, length, _ = logits.shape
    bit_ce = F.cross_entropy(logits.reshape(batch, length, -1, 2).permute(0, 3, 1, 2),
                             target_bits, reduction='none').mean(-1)
    bit_ce = bit_ce.mul(weights).sum(-1).mean()
    pred = F.normalize(aux['csa_pred'].float(), dim=-1)
    target = F.normalize(aux['csa_projected_target'].float().detach(), dim=-1)
    alignment = reduce(1.0 - (pred * target).sum(-1))
    factor = min(1.0, (step + 1) / csa_align_warmup_steps) if csa_align_warmup_steps > 0 else 1.0
    loss = bit_ce + csa_weight * factor * alignment
    return loss, {'loss': loss.detach(), 'bit_ce': bit_ce.detach(),
                  'align': alignment.detach(), 'reconstruction': zero,
                  'align_factor': factor}
