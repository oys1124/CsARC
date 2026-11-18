"""CPU tests for CSA losses, checkpoint handling and the training data contract."""
import json
import io
from contextlib import redirect_stderr
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(2)

from infinity.models.csa import CSAHead, CSATargetAutoencoder, compute_csa_loss, token_weights
from infinity.models.infinity import Infinity
from infinity.models.bsq_vae.multiscale_bsq import BSQ
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from infinity.utils.checkpoint import (initialize_model, convert_state_names,
    save_training_checkpoint, read_checkpoint, rng_state, restore_rng)
from infinity.dataset.build_stage2_dataset import build_stage2_dataloader
from trainer import configure_stage
from train import parse_args, config_dict, validate_resume_config

SCHEDULE = [(1, 1, 1), (1, 2, 2), (1, 4, 4)]


def tiny_vae():
    quantizer = SimpleNamespace(lfq=BSQ(dim=4, codebook_size=16, new_quant=True, codebook_scale=.5),
                               z_interplote_down='area', z_interplote_up='trilinear')
    return SimpleNamespace(embed_dim=4, codebook_dim=4, vocab_size=16, quantizer=quantizer)


def tiny_model():
    return Infinity(tiny_vae(), text_channels=128, text_maxlen=8, embed_dim=128,
                    depth=2, num_heads=2, block_chunks=2, shared_aln=True,
                    cond_drop_rate=0, raw_scale_schedule=None, pad_to_multiplier=4,
                    rope2d_each_sa_layer=1, rope2d_normalized_by_hw=2,
                    pn='1M', train_h_div_w_list=[1.0], use_flex_attn=False,
                    csa_projected_dim=8, csa_head_hidden_dim=16,
                    csa_target_projector_hidden_dim=16)


def conditions():
    return (torch.randn(5, 128), torch.tensor([2, 3], dtype=torch.int32),
            torch.tensor([0, 2, 5], dtype=torch.int32), 3)


class TestCSA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)

    def test_autoencoder_normalization(self):
        model = CSATargetAutoencoder(4, 8, 16)
        result = model(torch.randn(2, 21, 4))
        torch.testing.assert_close(result['target'].norm(dim=-1), torch.ones(2, 21))
        self.assertEqual(result['recon'].shape, (2, 21, 4))

    def test_scale_weights_match_source_precision(self):
        weights = token_weights(SCHEDULE, 'cpu')
        self.assertEqual(weights.dtype, torch.float64)
        self.assertAlmostEqual(float(weights.sum()), 1.)
        self.assertAlmostEqual(float(weights[0, 0] / weights[0, -1]), 4.)

    def test_stage_losses_and_gradient_groups(self):
        for stage in ('target_ae', 'align'):
            model = tiny_model()
            configure_stage(model, stage, fsdp=True)  # Test actual loss gating, not requires_grad freezing.
            codes = torch.sign(torch.randn(2, 21, 4)) / 2
            bits = (codes > 0).long()
            logits, aux = model(conditions(), torch.randn(2, 20, 4), SCHEDULE,
                                return_csa=True, csa_target_BLC=codes,
                                csa_target_ae_only=stage == 'target_ae')
            loss, metrics = compute_csa_loss(logits, aux, codes, bits, SCHEDULE,
                stage, 499, csa_scale_max=1, csa_align_warmup_steps=1000)
            loss.backward()
            target_grad = [p.grad for n, p in model.named_parameters() if n.startswith('csa_target_projector.')]
            hidden_grad = [p.grad for n, p in model.named_parameters() if n.startswith('csa_head.')]
            main_grad = model.head.weight.grad
            if stage == 'target_ae':
                self.assertTrue(all(g is not None for g in target_grad))
                self.assertTrue(all(g is None for g in hidden_grad))
                self.assertIsNone(main_grad)
                self.assertEqual(float(metrics['bit_ce']), 0.)
            else:
                self.assertTrue(all(g is None for g in target_grad))
                self.assertTrue(all(g is not None for g in hidden_grad))
                self.assertIsNotNone(main_grad)
                self.assertEqual(metrics['align_factor'], .5)
                torch.testing.assert_close(loss, metrics['bit_ce'] + .03 * .5 * metrics['align'])
                self.assertEqual(float(metrics['reconstruction']), 0.)
            self.assertTrue(torch.isfinite(loss))

    def test_inactive_alignment_scale_excluded(self):
        target = torch.sign(torch.randn(1, 21, 4)) / 2
        pred = torch.randn(1, 21, 8, requires_grad=True)
        aux = {'csa_pred': pred, 'csa_projected_target': F.normalize(torch.randn(1, 21, 8), dim=-1)}
        loss, _ = compute_csa_loss(torch.randn(1, 21, 8), aux, target,
                                  (target > 0).long(), SCHEDULE, 'align', 999, csa_scale_max=1)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(pred.grad[:, 5:])), 0)
        self.assertGreater(int(torch.count_nonzero(pred.grad[:, :5])), 0)

    def test_no_auxiliary_path_at_inference(self):
        model = tiny_model().eval()
        def reject(*unused):
            raise AssertionError('Auxiliary module used during bare backbone forward')
        model.csa_head.register_forward_hook(reject)
        model.csa_target_projector.register_forward_hook(reject)
        self.assertEqual(model(conditions(), torch.randn(2, 20, 4), SCHEDULE).shape, (2, 21, 8))

    def test_checkpoint_loading_and_dimension_rejection(self):
        model = tiny_model()
        csa = {k: v.clone() for k, v in model.state_dict().items()}
        report = initialize_model(model, {'gpt': csa})
        self.assertEqual(report['missing_auxiliary'], [])
        self.assertEqual(report['ignored_legacy'], [])
        converted, ignored = convert_state_names(csa, backbone_only=True)
        self.assertEqual(ignored, [])
        self.assertEqual(set(converted), {
            k for k in model.state_dict() if not k.startswith(('csa_head.', 'csa_target_projector.'))
        })
        csa['csa_head.net.3.weight'] = torch.randn(16, 16)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            initialize_model(model, {'gpt': csa})

    def test_checkpoint_resume_and_rng(self):
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        model(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pth'
            save_training_checkpoint(path, model, optimizer, {'stage': 'align'}, 1, 0, 1)
            saved = read_checkpoint(path)
            expected = (torch.rand(3), np.random.rand(3))
            restore_rng(saved['rng_by_rank'][0])
            torch.testing.assert_close(torch.rand(3), expected[0])
            np.testing.assert_array_equal(np.random.rand(3), expected[1])
            clone = nn.Linear(4, 2)
            clone.load_state_dict(saved['model'])
            other = torch.optim.AdamW(clone.parameters(), lr=1.)
            other.load_state_dict(saved['optimizer'])
            for network, opt in ((model, optimizer), (clone, other)):
                opt.zero_grad(set_to_none=True)
                network(torch.ones(2, 4)).sum().backward()
                opt.step()
            for a, b in zip(model.parameters(), clone.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_memmap_data_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'shards').mkdir()
            image = root / 'image.png'
            Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(image)
            record = {'path': str(image), 'shard': 0, 'idx': 0}
            (root / 'samples.jsonl').write_text(json.dumps(record) + '\n')
            torch.save(torch.tensor([0]), root / 'samples_offsets.pt')
            np.ones((3, 8), dtype=np.float16).tofile(root / 'shards' / 'kv.bin')
            torch.save({'rows': 3, 'hidden_size': 8, 'dtype': 'fp16', 'kv_bin': 'kv.bin',
                        'offsets': torch.tensor([0, 3]), 'lens': torch.tensor([3])},
                       root / 'shards' / 'shard_000000.meta.pt')
            bundle = build_stage2_dataloader('proc_memmap', str(root), [8], 1, 0)
            images, (kv, lens, cu, maximum) = next(iter(bundle.dataloader))
            self.assertEqual(images.shape, (1, 3, 8, 8))
            self.assertEqual(float(images.max()), -1.)
            self.assertEqual(kv.shape, (3, 8))
            self.assertEqual(cu.tolist(), [0, 3])
            self.assertEqual(maximum, 3)

    def test_recipe_validation_and_overrides(self):
        args = parse_args(['--dry_run'])
        self.assertEqual(args.csa_projected_dim, 256)
        self.assertEqual(args.effective_lr, 6e-5)
        paper = Path(__file__).resolve().parents[1] / 'configs/paper_proj256_align.json'
        args = parse_args(['--config', str(paper), '--dry_run', '--base_lr', '1e-4',
                           '--lr_mode', 'legacy_scaled', '--batch_size', '4'])
        self.assertEqual(args.effective_lr, 1e-4 * 4 / 256)
        self.assertEqual(args.csa_scale_max, 12)

    def test_fixed_random_self_correction_without_mode_argument(self):
        vae = tiny_vae()
        raw = torch.randn(2, 4, 4, 4)
        args = SimpleNamespace(noise_apply_layers=1, noise_apply_strength=0.,
                               noise_apply_requant=True)
        clean_inputs, clean_bits, clean_codes = BitwiseSelfCorrection(vae, args).flip_requant(SCHEDULE, raw)
        args.noise_apply_strength = .3
        # Force every eligible first-scale bit to flip; no mode attribute exists.
        def zero_draw(*size, **kwargs):
            return torch.zeros(*size, **kwargs)
        with patch('infinity.models.bitwise_self_correction.np.random.randint', return_value=30), \
                patch('infinity.models.bitwise_self_correction.torch.rand', side_effect=zero_draw):
            noisy_inputs, noisy_bits, noisy_codes = BitwiseSelfCorrection(vae, args).flip_requant(SCHEDULE, raw)
        torch.testing.assert_close(noisy_inputs[:, :4], -clean_inputs[:, :4], rtol=0, atol=0)
        torch.testing.assert_close(noisy_bits[0], clean_bits[0], rtol=0, atol=0)
        torch.testing.assert_close(noisy_codes[0], clean_codes[0], rtol=0, atol=0)

    def test_removed_mode_and_all_recipes(self):
        folder = Path(__file__).resolve().parents[1] / 'configs'
        for path in sorted(folder.glob('*.json')):
            with self.subTest(config=path.name):
                args = parse_args(['--config', str(path), '--dry_run'])
                self.assertFalse(hasattr(args, 'flip_mode'))
                self.assertNotIn('flip_mode', config_dict(args))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--dry_run', '--flip_mode', 'none'])

    def test_legacy_random_resume_configuration(self):
        args = parse_args(['--dry_run'])
        previous = config_dict(args)
        validate_resume_config(previous, args)
        previous['flip_mode'] = 'random'
        validate_resume_config(previous, args)
        previous['flip_mode'] = 'none'
        with self.assertRaisesRegex(ValueError, 'non-random'):
            validate_resume_config(previous, args)


if __name__ == '__main__':
    unittest.main()
