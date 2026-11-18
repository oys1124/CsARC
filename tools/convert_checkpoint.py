#!/usr/bin/env python3
"""Inspect model weights or export CSA and backbone checkpoints."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infinity.utils.checkpoint import read_checkpoint, extract_model_state, convert_state_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--backbone_only', action='store_true')
    parser.add_argument('--inspect_only', action='store_true')
    args = parser.parse_args()
    if not args.inspect_only and args.output is None:
        parser.error('--output is required unless --inspect_only is supplied')
    if args.output and args.output.resolve() == args.input.resolve():
        parser.error('Choose a new output path; the source checkpoint is never overwritten')
    checkpoint = read_checkpoint(args.input)
    state, ignored = convert_state_names(extract_model_state(checkpoint), args.backbone_only)
    auxiliary = {k: list(v.shape) for k, v in state.items() if k.startswith('csa_')}
    report = {'parameter_tensors': len(state), 'ignored_legacy': ignored,
              'backbone_only': args.backbone_only, 'csa_shapes': auxiliary}
    print(json.dumps(report, indent=2))
    if not args.inspect_only:
        if args.output.exists():
            parser.error('Output already exists; choose a new output path')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch = __import__('torch')
        # clone avoids retaining oversized/shared FSDP storage in slim exports.
        exported = {key: value.detach().cpu().clone() for key, value in state.items()}
        torch.save({'format': 'csa-weights-v1', 'gpt': exported,
                    'csa_config': report}, args.output)


if __name__ == '__main__':
    main()
