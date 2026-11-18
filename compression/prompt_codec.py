"""Caption serialization and T5 normalization for HPRC evaluation."""
from __future__ import annotations

import json
import struct
import zlib
from typing import Any, Dict, List, Mapping

import numpy as np
import torch


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")



def encode_prompt_record(tokenizer, prompt: str, mode: str = "arith", max_len: int = 512):
    """Return (record_bytes, payload_bits, decoder_prompt)."""
    mode = str(mode or "none").lower()
    if mode == "arith":
        import torchac

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(max_len),
            add_special_tokens=True,
        )
        ids = [int(value) for value in encoded["input_ids"][0].tolist()]
        text = ",".join(map(str, ids))
        frequencies: Dict[str, int] = {}
        for char in text:
            if char.isdigit() or char == ",":
                frequencies[char] = frequencies.get(char, 0) + 1
        unique_chars = sorted(frequencies)
        total = sum(frequencies.values())
        if total:
            probabilities = [frequencies[char] / total for char in unique_chars]
            cdf_1d = torch.zeros(len(unique_chars) + 1, dtype=torch.float32)
            cdf_1d[1:] = torch.cumsum(torch.tensor(probabilities, dtype=torch.float32), dim=0)
            cdf_1d[-1] = 1.0
            cdf = cdf_1d.view(1, 1, -1).expand(1, len(text), -1).contiguous()
            symbols = torch.tensor(
                [unique_chars.index(char) for char in text], dtype=torch.int16
            ).view(1, -1)
            payload = torchac.encode_float_cdf(cdf, symbols, check_input_bounds=True)
        else:
            cdf_1d = torch.tensor([0.0, 1.0], dtype=torch.float32)
            payload = b""

        cdf_bytes = cdf_1d.contiguous().numpy().astype("<f4", copy=False).tobytes()
        prompt_meta = {
            "mode": "arith",
            "unique_chars": unique_chars,
            "char_len": len(text),
            "ids_len": len(ids),
            "cdf_nbytes": len(cdf_bytes),
            "payload_nbytes": len(payload),
        }
        meta_bytes = _json_bytes(prompt_meta)
        record = struct.pack("<I", len(meta_bytes)) + meta_bytes + cdf_bytes + payload
        recovered = decode_prompt_record(record, tokenizer)
        return record, len(payload) * 8, recovered

    if mode == "zlib":
        payload = zlib.compress(prompt.encode("utf-8"))
        prompt_meta = {"mode": "zlib", "payload_nbytes": len(payload)}
        meta_bytes = _json_bytes(prompt_meta)
        return struct.pack("<I", len(meta_bytes)) + meta_bytes + payload, len(payload) * 8, prompt

    if mode == "none":
        payload = prompt.encode("utf-8")
        prompt_meta = {"mode": "raw_utf8", "payload_nbytes": len(payload)}
        meta_bytes = _json_bytes(prompt_meta)
        return struct.pack("<I", len(meta_bytes)) + meta_bytes + payload, 0, prompt

    raise ValueError(f"unsupported prompt mode: {mode!r}")



def decode_prompt_record(record: bytes, tokenizer) -> str:
    if len(record) < 4:
        raise ValueError("truncated prompt record")
    meta_len = struct.unpack_from("<I", record, 0)[0]
    meta_end = 4 + meta_len
    if meta_end > len(record):
        raise ValueError("truncated prompt metadata")
    meta = json.loads(record[4:meta_end].decode("utf-8"))
    mode = str(meta.get("mode", ""))

    if mode == "arith":
        import torchac

        cdf_nbytes = int(meta["cdf_nbytes"])
        payload_nbytes = int(meta["payload_nbytes"])
        cdf_end = meta_end + cdf_nbytes
        payload_end = cdf_end + payload_nbytes
        if payload_end != len(record):
            raise ValueError("invalid arithmetic prompt record length")
        cdf_1d = torch.from_numpy(
            np.frombuffer(record[meta_end:cdf_end], dtype="<f4").copy()
        )
        char_len = int(meta["char_len"])
        if char_len == 0:
            ids: List[int] = []
        else:
            cdf = cdf_1d.view(1, 1, -1).expand(1, char_len, -1).contiguous()
            symbols = torchac.decode_float_cdf(cdf, record[cdf_end:payload_end]).view(-1).tolist()
            unique_chars = list(meta["unique_chars"])
            text = "".join(unique_chars[int(index)] for index in symbols)
            ids = [int(value) for value in text.split(",") if value]
        if len(ids) != int(meta["ids_len"]):
            raise ValueError("decoded prompt ID count does not match stream metadata")
        if not ids:
            return ""
        try:
            return tokenizer.decode(ids, skip_special_tokens=True).strip()
        except TypeError:
            return tokenizer.decode(ids)

    payload_nbytes = int(meta.get("payload_nbytes", -1))
    payload = record[meta_end:]
    if payload_nbytes != len(payload):
        raise ValueError("invalid prompt payload length")
    if mode == "zlib":
        return zlib.decompress(payload).decode("utf-8")
    if mode == "raw_utf8":
        return payload.decode("utf-8")
    raise ValueError(f"unsupported prompt record mode: {mode!r}")

