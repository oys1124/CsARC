"""Bit-level selection utilities for progressive target-bpp rate control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from codec.arithmetic import TOT_FREQ


_EPS = 1e-6


@dataclass
class BitScaleSelectionData:
    scale_index: int
    height: int
    width: int
    bit_depth: int
    p1: torch.Tensor  # [1, L, d]
    rate_per_bit: torch.Tensor  # [L * d]
    score: torch.Tensor  # [L * d]

    @property
    def num_positions(self) -> int:
        return int(self.height) * int(self.width)

    @property
    def num_bits(self) -> int:
        return int(self.rate_per_bit.numel())

    @property
    def full_rate_bits(self) -> float:
        return float(self.rate_per_bit.sum().item())

    def all_bits(self) -> torch.Tensor:
        return torch.arange(self.num_bits, device=self.rate_per_bit.device, dtype=torch.long)

    def select_under_budget(
        self,
        budget_bits: float,
        *,
        tolerance_bits: float = 1e-6,
        descending: bool = True,
        order: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        budget = float(max(0.0, budget_bits))
        if budget <= 0.0 or self.num_bits == 0:
            return torch.zeros((0,), device=self.rate_per_bit.device, dtype=torch.long)
        if order is None:
            order = torch.argsort(self.score, descending=bool(descending))
        else:
            order = order.to(device=self.rate_per_bit.device, dtype=torch.long)
        ordered_rates = self.rate_per_bit.index_select(0, order)
        keep_mask = torch.cumsum(ordered_rates, dim=0) <= (budget + float(tolerance_bits))
        if not bool(keep_mask.any().item()):
            return torch.zeros((0,), device=self.rate_per_bit.device, dtype=torch.long)
        return order[keep_mask].contiguous()


def logits_to_p1(logits: torch.Tensor, *, bit_depth: int) -> torch.Tensor:
    batch, seq_len, vocab = logits.shape
    if vocab != int(bit_depth) * 2:
        raise ValueError(f"expected logits vocab={int(bit_depth) * 2}, got {vocab}")
    return logits.reshape(batch, seq_len, int(bit_depth), 2).softmax(dim=-1)[..., 1]


def build_bit_scale_selection_data(
    *,
    logits: torch.Tensor,
    scale_index: int,
    patch_shape,
    bit_depth: int,
    fixed_hw,
    ms: str,
    params: Mapping,
) -> BitScaleSelectionData:
    _, height, width = tuple(int(value) for value in patch_shape)
    depth = int(bit_depth)
    prob_tau = float(params.get("prob_tau", 0.5))
    if prob_tau <= 0:
        raise ValueError(f"prob_tau must be positive, got {prob_tau}")
    p_eff = logits_to_p1(logits.mul(1.0 / prob_tau), bit_depth=depth).float().clamp(_EPS, 1.0 - _EPS)
    freq1 = torch.floor(p_eff * float(TOT_FREQ) + 0.5).clamp(1, TOT_FREQ - 1)
    coder_p1 = (freq1 / float(TOT_FREQ)).clamp(_EPS, 1.0 - _EPS)
    rate_bits = -(
        p_eff * torch.log2(coder_p1) + (1.0 - p_eff) * torch.log2(1.0 - coder_p1)
    )
    budget_rate_scale = float(params.get("budget_rate_scale", 1.0))
    if budget_rate_scale <= 0:
        raise ValueError(f"budget_rate_scale must be positive, got {budget_rate_scale}")
    rate_per_bit = (rate_bits[0].reshape(-1) * budget_rate_scale).contiguous()

    ms = str(ms).lower()
    if ms == "entropy":
        entropy = -(p_eff * torch.log2(p_eff) + (1.0 - p_eff) * torch.log2(1.0 - p_eff))
        score = entropy[0].reshape(-1).contiguous()
    elif ms == "rdproxy":
        uncertainty = torch.minimum(p_eff, 1.0 - p_eff)
        scale_weight_mode = str(params.get("scale_weight", "area")).lower()
        if scale_weight_mode in ("none", "1", "unity"):
            scale_weight = 1.0
        else:
            area_weight = float(int(fixed_hw[0]) * int(fixed_hw[1])) / max(
                1.0, float(height * width)
            )
            scale_weight = math.sqrt(area_weight) if scale_weight_mode == "sqrt" else area_weight
        distortion_proxy = uncertainty[0].reshape(-1) * scale_weight * (4.0 / max(1.0, float(depth)))
        score = (distortion_proxy / (rate_per_bit + _EPS)).contiguous()
    else:
        raise ValueError(f"unsupported progressive ms={ms!r}; expected rdproxy or entropy")

    return BitScaleSelectionData(
        scale_index=int(scale_index),
        height=int(height),
        width=int(width),
        bit_depth=depth,
        p1=p_eff,
        rate_per_bit=rate_per_bit,
        score=score,
    )
