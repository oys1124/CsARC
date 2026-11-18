"""Incremental Infinity inference with persistent per-scale KV caches.

Supports batch size one, text conditioning, bit labels and CFG on logits.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from infinity.models.basic import CrossAttnBlock
from infinity.models.sampling import sample_with_top_k_top_p_also_inplace_modifying_logits_



ScaleSchedule = List[Tuple[int, int, int]]


@dataclass
class SessionOutput:
    """Output of one scale prediction before its bits are committed."""

    scale_index: int
    patch_shape: Tuple[int, int, int]
    logits: torch.Tensor  # [1, L, 2*d], conditional logits for entropy coding
    sample_logits: Optional[torch.Tensor] = None  # [1, L, 2*d], CFG/tau logits for sampling


def _infer_transformer_autocast_dtype(model: torch.nn.Module) -> Optional[torch.dtype]:
    """Return the dtype used by transformer attention weights, if half precision."""
    model = getattr(model, "_orig_mod", model)
    for chunk in getattr(model, "block_chunks", []):
        for module in getattr(chunk, "module", []):
            attn = getattr(module, "sa", getattr(module, "attn", None))
            mat_qkv = getattr(attn, "mat_qkv", None)
            weight = getattr(mat_qkv, "weight", None)
            if weight is not None and weight.dtype in (torch.bfloat16, torch.float16):
                return weight.dtype
    for module in getattr(model, "unregistered_blocks", []):
        attn = getattr(module, "sa", getattr(module, "attn", None))
        mat_qkv = getattr(attn, "mat_qkv", None)
        weight = getattr(mat_qkv, "weight", None)
        if weight is not None and weight.dtype in (torch.bfloat16, torch.float16):
            return weight.dtype
    return None


def _autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


class InfinityARSession:
    """Incremental Infinity inference with persistent self-attention KV caches."""

    def __init__(
        self,
        model: torch.nn.Module,
        vae: torch.nn.Module,
        scale_schedule: ScaleSchedule,
        text_condition: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
        *,
        seed: Optional[int] = None,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 0.0,
        empty_cache_before_vae: bool = True,
        decode_cfg: float = 1.0,
        decode_tau: float = 1.0,
        cfg_insertion_layer: Optional[List[int]] = None,
    ) -> None:
        self.model = getattr(model, "_orig_mod", model)
        self.vae = vae
        self.scale_schedule = [tuple(int(v) for v in pn) for pn in scale_schedule]
        self.do_sample = bool(do_sample)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.empty_cache_before_vae = bool(empty_cache_before_vae)
        self.decode_cfg_list = self._expand_float_list(decode_cfg, len(self.scale_schedule))
        self.decode_tau_list = self._expand_float_list(decode_tau, len(self.scale_schedule))
        self.cfg_insertion_layer = list(cfg_insertion_layer or [0])
        if any(item != 0 for item in self.cfg_insertion_layer):
            raise NotImplementedError("InfinityARSession currently supports CFG on logits only: [0]")
        self.add_cfg_on_logits = 0 in self.cfg_insertion_layer
        self.base_batch = 1
        self.batch_size = 2 if any(value != 1.0 for value in self.decode_cfg_list) else 1
        self.scale_index = 0
        self.summed_codes: Optional[torch.Tensor] = None
        self._pending: Optional[SessionOutput] = None
        self._closed = False
        self._kv_cache_enabled = False
        self._autocast_dtype = _infer_transformer_autocast_dtype(self.model)

        if not bool(getattr(self.model, "inference_mode", False)):
            raise ValueError("InfinityARSession requires a model built with inference_mode=True")
        if not bool(getattr(self.model, "use_bit_label", False)):
            raise NotImplementedError("InfinityARSession currently requires use_bit_label=True")
        if not bool(getattr(self.model, "t2i", False)):
            raise NotImplementedError("InfinityARSession currently requires text conditioning")
        if getattr(self.model, "condition_adapter", None) is not None:
            raise NotImplementedError("condition_adapter is not supported by the ARPC streaming session")
        if not hasattr(self.model, "block_chunks"):
            raise NotImplementedError("streaming inference currently requires block_chunks > 1")
        if not self.scale_schedule:
            raise ValueError("scale_schedule must not be empty")

        self.vae_scale_schedule = (
            [(pt, 2 * ph, 2 * pw) for pt, ph, pw in self.scale_schedule]
            if bool(getattr(self.model, "apply_spatial_patchify", False))
            else list(self.scale_schedule)
        )

        kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_condition
        if int(lens.shape[0]) != 1:
            raise ValueError(f"InfinityARSession requires B=1, got B={int(lens.shape[0])}")
        if self.batch_size == 2:
            kv_compact_un = kv_compact.clone()
            total = 0
            for length in lens:
                length = int(length)
                kv_compact_un[total : total + length] = self.model.cfg_uncond[:length]
                total += length
            kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
            cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0)

        kv_compact = self.model.text_norm(kv_compact)
        self.cond_BD = self.model.text_proj_for_sos(
            (kv_compact, cu_seqlens_k, max_seqlen_k)
        )
        ca_compact = self.model.text_proj_for_ca(kv_compact)
        self.ca_kv = (ca_compact, cu_seqlens_k, max_seqlen_k)
        self.last_stage = (
            self.cond_BD.unsqueeze(1).expand(self.batch_size, 1, -1)
            + self.model.pos_start.expand(self.batch_size, 1, -1)
        )
        with torch.amp.autocast("cuda", enabled=False):
            self.cond_BD_or_gss = self.model.shared_ada_lin(
                self.cond_BD.float()
            ).float().contiguous()

        self.rng: Optional[torch.Generator] = None
        self.set_seed(seed)

        self._set_kv_cache(True)

    @staticmethod
    def _expand_float_list(value, length: int) -> List[float]:
        if isinstance(value, (list, tuple)):
            if len(value) < int(length):
                raise ValueError(f"expected at least {length} values, got {len(value)}")
            return [float(item) for item in value[: int(length)]]
        return [float(value)] * int(length)

    def set_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            self.rng = None
            return
        device = next(self.model.parameters()).device
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(int(seed))

    def _set_kv_cache(self, enabled: bool) -> None:
        for block in self.model.unregistered_blocks:
            attn = block.sa if isinstance(block, CrossAttnBlock) else block.attn
            attn.kv_caching(bool(enabled))
        self._kv_cache_enabled = bool(enabled)

    def release_transformer_state(self) -> None:
        """Release autoregressive state before the memory-heavy VAE decoder."""
        if self._kv_cache_enabled:
            self._set_kv_cache(False)
        self.last_stage = None
        self.ca_kv = None
        self.cond_BD_or_gss = None
        self._pending = None

    @torch.no_grad()
    def predict_next_scale(self) -> SessionOutput:
        if self._closed:
            raise RuntimeError("session is closed")
        if self._pending is not None:
            raise RuntimeError("commit_scale() must be called before predicting another scale")
        if self.scale_index >= len(self.scale_schedule):
            raise StopIteration("all scales have already been predicted")

        si = int(self.scale_index)
        pn = self.scale_schedule[si]
        attn_fn = None
        if bool(getattr(self.model, "use_flex_attn", False)):
            attn_fn = self.model.attn_fn_compile_dict.get(
                tuple(self.scale_schedule[: si + 1]), None
            )

        hidden = self.last_stage
        with _autocast_context(hidden.device, self._autocast_dtype):
            for block_idx, chunk in enumerate(self.model.block_chunks):
                if self.model.add_lvl_embeding_only_first_block and block_idx == 0:
                    hidden = self.model.add_lvl_embeding(
                        hidden, si, self.scale_schedule, need_to_pad=0
                    )
                if not self.model.add_lvl_embeding_only_first_block:
                    hidden = self.model.add_lvl_embeding(
                        hidden, si, self.scale_schedule, need_to_pad=0
                    )
                for module in chunk.module:
                    hidden = module(
                        x=hidden,
                        cond_BD=self.cond_BD_or_gss,
                        ca_kv=self.ca_kv,
                        attn_bias_or_two_vector=None,
                        attn_fn=attn_fn,
                        scale_schedule=self.scale_schedule,
                        rope2d_freqs_grid=self.model.rope2d_freqs_grid,
                        scale_ind=si,
                    )

            logits_fn = getattr(self.model, "get_entropy_logits", None)
            if logits_fn is None:
                entropy_logits_all = self.model.get_logits(hidden, self.cond_BD)
            else:
                entropy_logits_all = logits_fn(hidden, self.cond_BD, scale_ind=si)
            coding_logits = entropy_logits_all[: self.base_batch]
            cfg = float(self.decode_cfg_list[si])
            tau = float(self.decode_tau_list[si])
            if tau <= 0:
                raise ValueError(f"decode_tau must be positive, got {tau}")
            if cfg != 1.0 and self.add_cfg_on_logits:
                sample_logits = cfg * entropy_logits_all[: self.base_batch] + (1.0 - cfg) * entropy_logits_all[self.base_batch :]
            else:
                sample_logits = coding_logits
            sample_logits = sample_logits.mul(1.0 / tau)

        batch, seq_len = coding_logits.shape[:2]
        pair_logits = coding_logits.reshape(batch, seq_len, -1, 2)
        sample_pair_logits = sample_logits.reshape(batch, seq_len, -1, 2)

        output = SessionOutput(
            scale_index=si,
            patch_shape=pn,
            logits=pair_logits.reshape(batch, seq_len, -1),
            sample_logits=sample_pair_logits.reshape(batch, seq_len, -1),
        )
        self._pending = output
        return output

    @torch.no_grad()
    def sample_current_scale(self, output: Optional[SessionOutput] = None) -> torch.Tensor:
        """Sample all current bits, matching Infinity's normal RNG consumption."""
        output = self._pending if output is None else output
        if output is None or output is not self._pending:
            raise RuntimeError("sample_current_scale() requires the current pending output")

        pt, ph, pw = output.patch_shape
        if pt != 1:
            raise NotImplementedError("streaming ARPC currently supports image scales with pt=1")
        logits = output.sample_logits if output.sample_logits is not None else output.logits
        batch, seq_len, vocab = logits.shape
        bit_depth = vocab // 2
        pair_logits = logits.reshape(batch, seq_len, bit_depth, 2)
        if self.do_sample:
            sampled = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                pair_logits.reshape(batch, -1, 2).clone(),
                rng=self.rng,
                top_k=self.top_k or int(getattr(self.model, "top_k", 0)),
                top_p=self.top_p or float(getattr(self.model, "top_p", 0.0)),
                num_samples=1,
            )[:, :, 0].reshape(batch, seq_len, bit_depth)
        else:
            sampled = pair_logits.argmax(dim=-1)

        sampled = sampled.reshape(batch, ph, pw, bit_depth)
        if bool(getattr(self.model, "apply_spatial_patchify", False)):
            sampled = F.pixel_shuffle(sampled.permute(0, 3, 1, 2), 2).permute(
                0, 2, 3, 1
            )
        return sampled.unsqueeze(1).to(torch.uint8)

    @torch.no_grad()
    def commit_scale(self, bit_indices: torch.Tensor) -> None:
        if self._closed:
            raise RuntimeError("session is closed")
        if self._pending is None:
            raise RuntimeError("predict_next_scale() must be called before commit_scale()")

        si = int(self.scale_index)
        bits = bit_indices
        if bits.ndim == 4:
            bits = bits.unsqueeze(1)
        if bits.ndim != 5 or int(bits.shape[0]) != 1 or int(bits.shape[1]) != 1:
            raise ValueError(f"expected bit indices [1,1,H,W,d], got {tuple(bits.shape)}")
        bits = bits.to(device=self.last_stage.device, dtype=torch.uint8)

        codes = self.vae.quantizer.lfq.indices_to_codes(bits, label_type="bit_label")
        last_scale = len(self.scale_schedule) - 1
        if si != last_scale:
            contribution = F.interpolate(
                codes,
                size=self.vae_scale_schedule[-1],
                mode=self.vae.quantizer.z_interplote_up,
            )
            self.summed_codes = (
                contribution
                if self.summed_codes is None
                else self.summed_codes + contribution
            )
            next_stage = F.interpolate(
                self.summed_codes,
                size=self.vae_scale_schedule[si + 1],
                mode=self.vae.quantizer.z_interplote_up,
            ).squeeze(-3)
            if bool(getattr(self.model, "apply_spatial_patchify", False)):
                next_stage = F.pixel_unshuffle(next_stage, 2)
            next_stage = next_stage.reshape(*next_stage.shape[:2], -1).permute(0, 2, 1)
            self.last_stage = self.model.word_embed(self.model.norm0_ve(next_stage))
            if self.batch_size != self.base_batch:
                self.last_stage = self.last_stage.repeat(self.batch_size // self.base_batch, 1, 1)
        else:
            self.summed_codes = codes if self.summed_codes is None else self.summed_codes + codes

        self.scale_index += 1
        self._pending = None

    @torch.no_grad()
    def finish_image(self) -> torch.Tensor:
        if self.scale_index != len(self.scale_schedule) or self._pending is not None:
            raise RuntimeError("all predicted scales must be committed before finish_image()")
        if self.summed_codes is None:
            raise RuntimeError("session has no accumulated codes")
        self.release_transformer_state()
        if self.empty_cache_before_vae and self.summed_codes.is_cuda:
            # KV tensors are large, layer-sized allocations. Return their freed
            # blocks before the VAE asks for multi-hundred-MiB feature maps.
            torch.cuda.empty_cache()
        image = self.vae.decode(self.summed_codes.squeeze(-3))
        image = (image + 1) / 2
        return image.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))

    def close(self) -> None:
        if not self._closed:
            if self._kv_cache_enabled:
                self._set_kv_cache(False)
            self._closed = True

    def __enter__(self) -> "InfinityARSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
