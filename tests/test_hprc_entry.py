"""HPRC CLI defaults, decoder integration and evaluation resume."""
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

import eval_hprc as entry
from compression import rc_progressive_codec_bit_level as codec
from infinity.models.infinity import Infinity
from test_inference import TinyVAE, SCHEDULE


class TestHPRCEntry(unittest.TestCase):
    required = ['--model_path', '/not-loaded/model.pth', '--vae_path', '/not-loaded/vae.pth',
                '--text_encoder_ckpt', '/not-loaded/t5', '--out_dir', '/not-created/output']

    def test_effective_defaults_and_direct_two_pass_wiring(self):
        before = list(sys.argv)
        args = entry.build_parser().parse_args(self.required)
        self.assertEqual(sys.argv, before)
        self.assertEqual((args.decode_cfg, args.decode_tau, args.prob_tau,
                          args.budget_rate_scale, args.rate_overhead_bits),
                         (3., .5, .5, 1.6, 256.))
        entry._validate_decode_policy(args)
        self.assertIs(entry.decode_progressive_rate_control, codec.decode_progressive_rate_control)
        self.assertIs(entry.encode_progressive_rate_control, codec.encode_progressive_rate_control)
        self.assertNotIn('_runner', vars(entry))

    def test_explicit_overrides_and_cfg_values(self):
        args = entry.build_parser().parse_args(self.required + [
            '--decode_cfg=1', '--decode_tau', '.8', '--prob_tau=.7',
            '--budget_rate_scale', '1.2', '--rate_overhead_bits=12'])
        self.assertEqual((args.decode_cfg, args.decode_tau, args.prob_tau,
                          args.budget_rate_scale, args.rate_overhead_bits),
                         (1., .8, .7, 1.2, 12.))
        for value in (1., 3., 7.5):
            args.decode_cfg = value
            entry._validate_decode_policy(args)

    def test_invalid_probability_and_temperature_policy(self):
        args = entry.build_parser().parse_args(self.required)
        for field, value in [('decode_tau', 0.), ('decode_tau', float('nan')),
                             ('prob_tau', -1.), ('budget_rate_scale', float('inf')),
                             ('decode_cfg', float('nan'))]:
            invalid = SimpleNamespace(**vars(args))
            setattr(invalid, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                entry._validate_decode_policy(invalid)

    def test_dry_run_needs_no_models_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captions = root / 'captions.json'
            captions.write_text(json.dumps([{'img_path': '/not-read/image.png', 'txt': 'test'}]))
            out = root / 'out'
            argv = self.required + ['--out_dir', str(out), '--caption_json', str(captions),
                                    '--target_bpps', '.01,.02', '--dry_run', '1']
            with patch.object(entry, 'load_tokenizer', side_effect=AssertionError('model load during dry run')), \
                    patch.object(entry, 'load_visual_tokenizer', side_effect=AssertionError('VAE load during dry run')), \
                    redirect_stdout(io.StringIO()):
                entry.main(argv)
            self.assertFalse(out.exists())

    def test_runner_writes_real_hprc_streams_and_resumes(self):
        # Exercise the entry's full encode -> file -> decode -> CSV/resume path.
        # External weights, preprocessing, metrics and summary aggregation are
        # substituted; coding, stream I/O, per-image CSV and resume are real.
        torch.manual_seed(31)
        vae = TinyVAE(32)
        model = Infinity(vae, text_channels=128, text_maxlen=8, embed_dim=128,
            depth=2, num_heads=2, block_chunks=2, shared_aln=True, cond_drop_rate=0,
            raw_scale_schedule=None, pad_to_multiplier=4, rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2, pn='1M', train_h_div_w_list=[1.0],
            use_flex_attn=False, inference_mode=True, enable_csa=False).eval()
        text = (torch.randn(3, 128), torch.tensor([3], dtype=torch.int32),
                torch.tensor([0, 3], dtype=torch.int32), 3)
        bits = [torch.randint(0, 2, (1, h*w, 32), dtype=torch.uint8) for _, h, w in SCHEDULE]
        raw = torch.randn(1, 32, 4, 4)
        vae.encode_for_raw_features = lambda *unused, **kw: (raw, None, None)
        q_schedule = [(*scale, 2) for scale in SCHEDULE]
        metrics_modules = {
            'cv2': SimpleNamespace(),
            'lpips': SimpleNamespace(LPIPS=lambda **kw: nn.Identity()),
            'DISTS_pytorch': SimpleNamespace(DISTS=nn.Identity),
            'pytorch_msssim': SimpleNamespace(ms_ssim=lambda *a, **kw: torch.tensor(1.)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captions = root / 'captions.json'
            captions.write_text(json.dumps([{'img_path': str(root / 'image.png'), 'txt': 'x'}]))
            out = root / 'out'
            argv = self.required + ['--out_dir', str(out), '--caption_json', str(captions),
                '--target_bpps', '5,8,16', '--prob_tau', '1', '--budget_rate_scale', '1',
                '--rate_overhead_bits', '0', '--save_recon', '0', '--keep_gpu_busy', '0',
                '--add_prompt_bits', '0', '--prompt_bits_mode', 'none']
            with patch.dict(sys.modules, metrics_modules), \
                    patch('torch.cuda.is_available', return_value=False), \
                    patch.object(entry, 'load_tokenizer', return_value=(None, None)), \
                    patch.object(entry, 'load_visual_tokenizer', return_value=vae), \
                    patch.object(entry, 'load_transformer', return_value=model) as load_model, \
                    patch.object(entry, 'load_img', return_value=torch.zeros(1, 3, 4, 4)), \
                    patch.object(entry, '_scale_schedule_for_image', return_value=(SCHEDULE, q_schedule)), \
                    patch.object(entry, 'mask_quant', return_value=(None, bits)), \
                    patch.object(entry, 'encode_prompt', return_value=text), \
                    patch.object(entry, '_compute_metrics', return_value=(20., .9, .1, .2)), \
                    patch.object(entry, 'write_summary'), \
                    patch.object(entry, 'decode_progressive_rate_control', wraps=codec.decode_progressive_rate_control) as decode, \
                    redirect_stdout(io.StringIO()):
                entry.main(argv)
                self.assertEqual(decode.call_count, 3)
                self.assertTrue(all(call.kwargs['params']['decode_cfg'] == 3. for call in decode.call_args_list))
                with (out / 'rate_control_progressive.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 3)
                self.assertEqual(len(list((out / 'streams').rglob('*.rcp'))), 3)
                self.assertTrue(all(float(row['decode_cfg']) == 3. for row in rows))
                self.assertTrue(all(int(row['target_header_bytes']) == 4 for row in rows))
                load_model.reset_mock()
                entry.main(argv + ['--resume', '1'])
                load_model.assert_not_called()
                with (out / 'rate_control_progressive.csv').open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 3)


if __name__ == '__main__':
    unittest.main()
