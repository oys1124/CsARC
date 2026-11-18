"""RangeCoder-only scale evaluation: CLI, files, prefix verification and rates."""
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import eval_scale_stream as entry
from compression.raw_stream_io_v3 import PROFILE_INFINITY_1024_SCHEDULE
from test_inference import profile_codec_fixture


class TestScaleEntry(unittest.TestCase):
    required = [
        '--model_path', '/not-loaded/model.pth', '--vae_path', '/not-loaded/vae.pth',
        '--text_encoder_ckpt', '/not-loaded/t5'
    ]

    def test_range_only_cli(self):
        parser = entry.build_parser()
        args = parser.parse_args(self.required)
        self.assertEqual((args.tau, args.decode_cfg, args.decode_tau), (1., 3., .5))
        self.assertEqual(args.rng_mode, 'prefix_hash')
        self.assertEqual(args.verify_stream, 0)
        self.assertFalse(hasattr(args, 'stream_format'))
        self.assertFalse(hasattr(args, 'raw_stream_mode'))
        self.assertFalse(hasattr(args, 'prompt_transport'))
        for removed in [['--stream_format', 'framed_v1'],
                        ['--stream_format', 'scale_arithmetic_v2'], ['--raw_stream_mode', 'memory'],
                        ['--rng_mode', 'legacy_rng'], ['--prompt_transport', 'shared'],
                        ['--prompt_transport', 'standalone_zlib']]:
            with self.subTest(arguments=removed), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                parser.parse_args(self.required + removed)

    def test_dry_run_without_models_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captions = root / 'captions.json'
            captions.write_text(json.dumps([{'img_path': '/not-read/image.png', 'txt': 'test'}]))
            out = root / 'out'
            with patch.object(entry, 'load_tokenizer', side_effect=AssertionError('model load')), \
                    patch.object(entry, 'load_visual_tokenizer', side_effect=AssertionError('VAE load')), \
                    redirect_stdout(io.StringIO()) as output:
                entry.main(self.required +
                           ['--caption_json',
                            str(captions), '--out_dir',
                            str(out), '--dry_run'])
            self.assertEqual(json.loads(output.getvalue())['stream_format'], 'zero_header_v3')
            self.assertFalse(out.exists())

    def test_file_evaluation_and_prefix_rate_accounting(self):
        (model, vae, text, gt, _) = profile_codec_fixture()
        image = torch.zeros(1, 3, 64, 64)
        vae.encode_for_raw_features = lambda *args, **kw: (torch.zeros(1, 32, 64, 64), None, None)
        cv2 = SimpleNamespace(imread=lambda unused: np.zeros((64, 64, 3), dtype=np.uint8),
                              cvtColor=lambda image, unused: image,
                              COLOR_BGR2RGB=0)
        metric = lambda *args, **kw: torch.tensor(0.25)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                entry, 'device', 'cpu',
                create=True), patch.object(entry, 'cv2', cv2, create=True), patch.object(
                    entry, 'ms_ssim', metric,
                    create=True), patch.object(entry, 'load_img', return_value=image), patch.object(
                        entry, 'mask_quant', return_value=(None, gt)), patch.object(
                            entry, 'decompress_cfg', return_value=image) as reconstruct, patch(
                                'compression.sender.encode_prompt',
                                return_value=text), patch('compression.reciever.encode_prompt',
                                                          return_value=text):
            schedule = PROFILE_INFINITY_1024_SCHEDULE
            for mode in ('multirate', 'prefix_only'):
                for rng in ('prefix_hash', 'stream_seed'):
                    with self.subTest(transport='embedded', mode=mode, rng=rng):
                        args = SimpleNamespace(vae_type=32,
                                               tau=1.0,
                                               decode_cfg=3.0,
                                               decode_tau=0.5,
                                               stream_fsync=0,
                                               save_recon=0,
                                               rng_mode=rng,
                                               hash_seed_salt='CSA-test',
                                               decode_seed_base=0)
                        path = str(Path(tmp) / 'image.iraw')
                        boundary = 2 if mode == 'prefix_only' else None
                        (encoded, _, encode_times) = entry.compress_to_file(args,
                                                                            vae,
                                                                            schedule, [],
                                                                            model,
                                                                            None,
                                                                            None,
                                                                            '/not-read/image.png',
                                                                            'test',
                                                                            path,
                                                                            'test',
                                                                            1024,
                                                                            1024,
                                                                            max_scale=boundary)
                        reconstruct.reset_mock()
                        (rows,
                         decode_times) = entry.decompress_modified_from_file(args,
                                                                             model,
                                                                             vae,
                                                                             schedule,
                                                                             None,
                                                                             None,
                                                                             tmp,
                                                                             '/not-read/image.png',
                                                                             'image.png',
                                                                             path,
                                                                             'cpu',
                                                                             1024,
                                                                             1024,
                                                                             metric,
                                                                             metric,
                                                                             expected_prompt='test',
                                                                             expected_gt=gt,
                                                                             max_scale=boundary,
                                                                             only_scale=boundary)
                        expected_scales = ['2'] if boundary is not None else list(
                            map(str, range(13)))
                        self.assertEqual(list(rows), expected_scales)
                        self.assertEqual(reconstruct.call_count, len(expected_scales))
                        self.assertIn('encode_total_s', encode_times)
                        self.assertIn('range_decode_s', decode_times)
                        for (scale, row) in rows.items():
                            index = int(scale)
                            self.assertEqual(row['stream_prefix_bytes'],
                                             encoded.stream_prefix_bytes[index])
                            self.assertEqual(row['bpp'],
                                             encoded.stream_prefix_bytes[index] * 8 / 1024**2)
                            self.assertEqual(row['coding_payload_bpp'],
                                             encoded.coding_prefix_bits[index] / 1024**2)
                            self.assertEqual(row['standalone_equivalent_bpp'], row['stream_bpp'])
                            if rng == 'stream_seed':
                                self.assertEqual(row['decode_seed'], 0)
                        invalid_gt = [bits.clone() for bits in gt]
                        invalid_gt[0][0, 0, 0] ^= 1
                        with self.assertRaisesRegex(AssertionError, 'entropy decode mismatch'):
                            entry.decompress_modified_from_file(args,
                                                                model,
                                                                vae,
                                                                schedule,
                                                                None,
                                                                None,
                                                                tmp,
                                                                '/not-read/image.png',
                                                                'image.png',
                                                                path,
                                                                'cpu',
                                                                1024,
                                                                1024,
                                                                metric,
                                                                metric,
                                                                expected_prompt='test',
                                                                expected_gt=invalid_gt,
                                                                max_scale=boundary)


if __name__ == '__main__':
    unittest.main()
