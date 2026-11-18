"""CPU round trips for the two inference paths and shared stream primitives."""
import unittest
import tempfile
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)
from codec.arithmetic_accel import RangeEncoder, RangeDecoder
from compression.rc_progressive_codec_bit_level import encode_progressive_rate_control, decode_progressive_rate_control
from compression.rc_progressive_stream import pack_progressive_target_stream, unpack_progressive_target_stream
from compression.rc_progressive_selector_bit_level import build_bit_scale_selection_data
from compression.raw_stream_io_v3 import build_zero_header_stream, parse_zero_header_stream
from compression.raw_stream_io_v3 import MODE_ARITHMETIC, PROFILE_INFINITY_1024_SCHEDULE
from compression.sender import get_prob
from compression.reciever import decompress_cfg
from compression.sender import encoding_to_stream
from compression.reciever import decoding_from_stream
from infinity.models.infinity import Infinity
from infinity.models.infinity_ar_session import InfinityARSession
from infinity.models.bsq_vae.multiscale_bsq import BSQ

SCHEDULE = [(1, 1, 1), (1, 2, 2), (1, 4, 4)]


class TinyVAE(nn.Module):

    def __init__(self, depth=4):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.embed_dim = self.codebook_dim = depth
        self.vocab_size = 2**depth
        self.quantizer = SimpleNamespace(lfq=BSQ(dim=depth,
                                                 codebook_size=2**depth,
                                                 new_quant=True,
                                                 codebook_scale=depth**-.5),
                                         z_interplote_down='area',
                                         z_interplote_up='trilinear')

    def decode(self, z):
        return z[:, :3].clamp(-1, 1)


def tiny_codec():
    torch.manual_seed(73)
    vae = TinyVAE()
    model = Infinity(vae,
                     text_channels=128,
                     text_maxlen=8,
                     embed_dim=128,
                     depth=2,
                     num_heads=2,
                     block_chunks=2,
                     shared_aln=True,
                     cond_drop_rate=0,
                     raw_scale_schedule=None,
                     pad_to_multiplier=4,
                     rope2d_each_sa_layer=1,
                     rope2d_normalized_by_hw=2,
                     pn='1M',
                     train_h_div_w_list=[1.0],
                     use_flex_attn=False,
                     inference_mode=True,
                     enable_csa=False).eval()
    text = (torch.randn(3, 128), torch.tensor([3], dtype=torch.int32),
            torch.tensor([0, 3], dtype=torch.int32), 3)
    gt = [torch.randint(0, 2, (1, h * w, 4), dtype=torch.uint8) for _, h, w in SCHEDULE]
    return model, vae, text, gt


def profile_codec_fixture():

    class Block(nn.Module):

        def __init__(self):
            super().__init__()
            self.attn = SimpleNamespace(kv_caching=lambda enabled: None)

        def forward(self, x, **unused):
            return x

    class Pool(nn.Module):

        def forward(self, condition):
            return condition[0].new_zeros(1, 32)

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.text_norm = self.text_proj_for_ca = nn.Identity()
            self.text_proj_for_sos = Pool()
            self.shared_ada_lin = self.word_embed = self.norm0_ve = nn.Identity()
            self.pos_start = torch.zeros(1, 1, 32)
            self.use_flex_attn = False
            self.add_lvl_embeding_only_first_block = True
            block = Block()
            self.unregistered_blocks = [block]
            self.block_chunks = [SimpleNamespace(module=[block])]
            self.rope2d_freqs_grid = None
            self.rng = torch.Generator()
            p = torch.linspace(.05, .95, 32)
            self.logits = torch.stack((1 - p, p), -1).log().reshape(1, 1, 64)

        def add_lvl_embeding(self, x, *args, **kwargs):
            return x

        def get_entropy_logits(self, x, *args, **kwargs):
            return self.logits.expand(1, x.shape[1], 64)

    model, vae = Model(), TinyVAE(32)
    text = (torch.zeros(3, 32), [3], torch.tensor([0, 3], dtype=torch.int32), 3)
    random = torch.Generator().manual_seed(44)
    p = model.logits.reshape(32, 2).softmax(-1)[:, 1]
    gt = [(torch.rand(1, h * w, 32, generator=random) < p).to(torch.uint8)
          for _, h, w in PROFILE_INFINITY_1024_SCHEDULE]
    args = SimpleNamespace(vae_type=32, tau=1.)
    return model, vae, text, gt, args


class TestInference(unittest.TestCase):

    def test_real_thirteen_scale_stream_file_round_trip(self):
        (model, vae, text, gt, args) = profile_codec_fixture()
        with tempfile.TemporaryDirectory() as tmp, patch('compression.sender.encode_prompt',
                                                         return_value=text), patch(
                                                             'compression.reciever.encode_prompt',
                                                             return_value=text) as decode_prompt:
            path = str(Path(tmp) / 'image.iraw')
            encoded = encoding_to_stream(args,
                                         model,
                                         vae,
                                         PROFILE_INFINITY_1024_SCHEDULE,
                                         'test',
                                         'test',
                                         None,
                                         None,
                                         gt,
                                         stream_path=path,
                                         width=1024,
                                         height=1024)
            decoded = decoding_from_stream(args,
                                           model,
                                           vae,
                                           None,
                                           None,
                                           stream_path=path,
                                           expected_scale_schedule=PROFILE_INFINITY_1024_SCHEDULE)
            self.assertEqual(len(decoded.decoded_indices), 13)
            self.assertEqual(decoded.prompt, 'test')
            self.assertEqual(decode_prompt.call_args.args[2], 'test')
            for (expected, actual) in zip(gt, decoded.decoded_indices):
                torch.testing.assert_close(actual.to(expected.dtype), expected, rtol=0, atol=0)
            self.assertEqual(decoded.stream.file_bytes, encoded.stream_file_bytes)
            self.assertEqual(set(encoded.scale_modes), {0, 1})
            prefix_path = Path(tmp) / 'prefix.iraw'
            prefix_path.write_bytes(Path(path).read_bytes()[:encoded.stream_prefix_bytes[2]])
            prefix = decoding_from_stream(args,
                                          model,
                                          vae,
                                          None,
                                          None,
                                          stream_path=str(prefix_path))
            self.assertEqual(len(prefix.decoded_indices), 3)
            for (expected, actual) in zip(gt, prefix.decoded_indices):
                torch.testing.assert_close(actual.to(expected.dtype), expected, rtol=0, atol=0)

    def test_range_coder_round_trip(self):
        random = np.random.default_rng(4)
        probabilities = random.uniform(.001, .999, 4096).astype(np.float32)
        bits = (random.random(4096) < probabilities).astype(np.uint8)
        encoder = RangeEncoder()
        encoder.encode(bits[:2000], probabilities[:2000])
        encoder.encode(bits[2000:], probabilities[2000:])
        payload = encoder.finish()
        np.testing.assert_array_equal(RangeDecoder(payload).decode(probabilities), bits)

    def test_caption_is_always_embedded_and_header_is_required(self):
        for prompt in ['', 'cat', '一张清晰的照片', 'a clear photo of a garden ' * 64]:
            with self.subTest(prompt=prompt[:24]):
                payload, prefixes = build_zero_header_stream(
                    width=1024, height=1024, bit_depth=32,
                    scale_schedule=PROFILE_INFINITY_1024_SCHEDULE, prompt=prompt,
                    scale_payloads=[], scale_modes=[])
                decoded = parse_zero_header_stream(payload, bit_depth=32)
                self.assertEqual(decoded.prompt, prompt)
                self.assertGreaterEqual(decoded.header_bytes, 2)
                self.assertEqual(decoded.header_bytes, len(payload))
                self.assertEqual(prefixes, [])
        for malformed in [b'', b'\x00', struct.pack('<H', 3) + b'a',
                          struct.pack('<H', 0x8001) + b'\xff', struct.pack('<H', 2) + b'xx']:
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                parse_zero_header_stream(malformed, bit_depth=32)

    def test_zero_header_embedded_caption_and_independent_prefix(self):
        (model, vae, text, gt, args) = profile_codec_fixture()
        with tempfile.TemporaryDirectory() as tmp, patch('compression.sender.encode_prompt',
                                                         return_value=text), patch(
                                                             'compression.reciever.encode_prompt',
                                                             return_value=text):
            path = str(Path(tmp) / 'prefix.iraw')
            encoded = encoding_to_stream(args,
                                         model,
                                         vae,
                                         PROFILE_INFINITY_1024_SCHEDULE,
                                         'test',
                                         'test',
                                         None,
                                         None,
                                         gt,
                                         stream_path=path,
                                         width=1024,
                                         height=1024,
                                         max_scale=5)
            decoded = decoding_from_stream(args,
                                           model,
                                           vae,
                                           None,
                                           None,
                                           stream_path=path,
                                           max_scale=5)
            self.assertEqual(len(encoded.stream_prefix_bytes), 6)
            self.assertEqual(len(decoded.decoded_indices), 6)
            self.assertGreater(decoded.stream.header_bytes, 0)
            for (expected, actual) in zip(gt, decoded.decoded_indices):
                torch.testing.assert_close(actual.to(expected.dtype), expected, rtol=0, atol=0)

    def test_scale_stream_container_and_prefix(self):
        payloads = [bytes([index, index + 1]) for index in range(13)]
        (packed, prefix) = build_zero_header_stream(width=1024,
                                                    height=1024,
                                                    bit_depth=32,
                                                    scale_schedule=PROFILE_INFINITY_1024_SCHEDULE,
                                                    prompt='a clear photo',
                                                    scale_payloads=payloads,
                                                    scale_modes=[MODE_ARITHMETIC] * 13)
        stream = parse_zero_header_stream(packed, bit_depth=32)
        self.assertEqual(stream.prompt, 'a clear photo')
        self.assertEqual(stream.scale_payloads, payloads)
        for boundary in [0, 5, 12]:
            part = parse_zero_header_stream(packed[:prefix[boundary]], bit_depth=32)
            self.assertEqual(part.scale_payloads, payloads[:boundary + 1])

    def test_hprc_target_header_round_trip(self):
        packed = pack_progressive_target_stream(prompt='CSA caption',
                                                payload=b'\x00\xff',
                                                target_payload_bits=117)
        result = unpack_progressive_target_stream(packed)
        self.assertEqual(result.prompt, 'CSA caption')
        self.assertEqual(result.payload, b'\x00\xff')
        self.assertEqual(result.target_payload_bits, 117)
        self.assertEqual(result.target_header_bytes, 4)

    def test_entropy_selection_is_bit_level(self):
        p = torch.tensor([.5, .1, .3, .01])
        logits = torch.stack((1 - p, p), -1).log().reshape(1, 1, 8)
        data = build_bit_scale_selection_data(logits=logits,
                                              scale_index=0,
                                              patch_shape=(1, 1, 1),
                                              bit_depth=4,
                                              fixed_hw=(4, 4),
                                              ms='entropy',
                                              params={
                                                  'prob_tau': 1.,
                                                  'budget_rate_scale': 1.
                                              })
        self.assertEqual(torch.argsort(data.score, descending=True).tolist(), [0, 2, 1, 3])
        kept = data.select_under_budget(float(data.rate_per_bit[0]) + 1e-5)
        self.assertEqual(kept.tolist(), [0])

    def test_session_matches_scale_wise_coding_probabilities(self):
        model, vae, text, gt = tiny_codec()
        import compression.sender as raw_sender
        with torch.no_grad(), patch.object(raw_sender, 'encode_prompt', return_value=text):
            probabilities = get_prob(model, vae, SCHEDULE, 'test', None, None, gt, tau_list=1.)
        with InfinityARSession(model,
                               vae,
                               SCHEDULE,
                               text,
                               do_sample=False,
                               decode_cfg=1.,
                               decode_tau=1.,
                               empty_cache_before_vae=False) as session:
            for si, (_, h, w) in enumerate(SCHEDULE):
                output = session.predict_next_scale()
                actual = output.logits.reshape(-1, 2).softmax(-1)
                torch.testing.assert_close(actual, probabilities[si], rtol=0, atol=0)
                session.commit_scale(gt[si].reshape(1, 1, h, w, 4))
            self.assertEqual(session.finish_image().shape, (1, 4, 4, 3))

    def test_scale_wise_cfg_generation(self):
        model, vae, text, gt = tiny_codec()
        with torch.no_grad(), patch('compression.reciever.encode_prompt', return_value=text):
            first = decompress_cfg(model,
                                   vae,
                                   SCHEDULE,
                                   'test',
                                   None,
                                   None,
                                   2,
                                   gt,
                                   cfg_list=3.,
                                   tau_list=.5)
            expected = 0
            for (_, h, w), bits in zip(SCHEDULE, gt):
                code = vae.quantizer.lfq.indices_to_codes(bits.reshape(1, 1, h, w, 4),
                                                          label_type='bit_label')
                expected = expected + torch.nn.functional.interpolate(
                    code, size=SCHEDULE[-1], mode='trilinear')
            expected = (vae.decode(expected.squeeze(2)) + 1) / 2
            torch.testing.assert_close(first, expected, rtol=0, atol=0)
            model.rng.manual_seed(19)
            mixed_a = decompress_cfg(model,
                                     vae,
                                     SCHEDULE,
                                     'test',
                                     None,
                                     None,
                                     0,
                                     gt,
                                     cfg_list=3.,
                                     tau_list=.5)
            model.rng.manual_seed(19)
            mixed_b = decompress_cfg(model,
                                     vae,
                                     SCHEDULE,
                                     'test',
                                     None,
                                     None,
                                     0,
                                     gt,
                                     cfg_list=3.,
                                     tau_list=.5)
            torch.testing.assert_close(mixed_a, mixed_b, rtol=0, atol=0)

    def test_two_pass_hprc_round_trip_and_seeds(self):
        model, vae, text, gt = tiny_codec()
        params = {
            'prob_tau': 1.,
            'decode_cfg': 3.,
            'decode_tau': .5,
            'budget_rate_scale': 1.,
            'rate_overhead_bits': 0.,
            'seed_policy': 'prefix_hash',
            'hash_seed_salt': 'CSA-test',
            'bit_select_order': 'high',
            'empty_cache_before_vae': False
        }
        counts = []
        for budget in [1, 10, 30, 200]:
            common = dict(model=model,
                          vae=vae,
                          scale_schedule=SCHEDULE,
                          text_condition=text,
                          prompt='CSA test',
                          target_bpp=budget / 16,
                          prompt_bits=0,
                          ms='entropy',
                          bit_depth=4,
                          fixed_hw=(4, 4),
                          params=params)
            encoded = encode_progressive_rate_control(**common, gt_ms_idx_Bl=gt)
            first = decode_progressive_rate_control(**common,
                                                    payload=encoded.payload,
                                                    gt_ms_idx_Bl=gt)
            second = decode_progressive_rate_control(**common, payload=encoded.payload)
            self.assertEqual(encoded.per_scale_kept_bits, first.per_scale_kept_bits)
            self.assertEqual(encoded.boundary_scale, first.boundary_scale)
            self.assertEqual(encoded.decode_seed, first.decode_seed)
            torch.testing.assert_close(first.image_bhwc_u8, second.image_bhwc_u8, rtol=0, atol=0)
            counts.append(sum(encoded.per_scale_kept_bits))
            if budget == 200:
                self.assertEqual(first.full_scales, 3)
                self.assertEqual(first.boundary_scale, -1)
                self.assertEqual(first.per_scale_kept_bits, [4, 16, 64])
        self.assertEqual(counts, sorted(counts))
        self.assertGreater(counts[2], counts[0])


if __name__ == '__main__':
    unittest.main()
