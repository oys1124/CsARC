"""Optional compiled range coder with an exact Python fallback."""

from __future__ import annotations

from typing import Any

import numpy as np

from codec.arithmetic import RangeDecoder as PythonRangeDecoder
from codec.arithmetic import RangeEncoder as PythonRangeEncoder

try:
    from codec import _arithmetic_fast as _fast
except ImportError:
    _fast = None


FAST_CODER_AVAILABLE = _fast is not None


class RangeEncoder:
    def __init__(self) -> None:
        self._impl: Any = _fast.RangeEncoder() if FAST_CODER_AVAILABLE else PythonRangeEncoder()

    def encode(self, bits: np.ndarray, probabilities: np.ndarray) -> None:
        bits = np.ascontiguousarray(bits, dtype=np.uint8).reshape(-1)
        probabilities = np.ascontiguousarray(probabilities, dtype=np.float32).reshape(-1)
        if bits.size != probabilities.size:
            raise ValueError("bits and probabilities must have the same size")
        if FAST_CODER_AVAILABLE:
            self._impl.encode(bits, probabilities)
        else:
            for bit, probability in zip(bits.tolist(), probabilities.tolist()):
                self._impl.encode_bit(int(bit), float(probability))

    def finish(self) -> bytes:
        return bytes(self._impl.finish())


class RangeDecoder:
    def __init__(self, payload: bytes) -> None:
        self._impl: Any = (
            _fast.RangeDecoder(payload) if FAST_CODER_AVAILABLE else PythonRangeDecoder(payload)
        )

    def decode(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.ascontiguousarray(probabilities, dtype=np.float32).reshape(-1)
        if FAST_CODER_AVAILABLE:
            return np.asarray(self._impl.decode(probabilities), dtype=np.uint8)
        output = np.empty((probabilities.size,), dtype=np.uint8)
        for index, probability in enumerate(probabilities.tolist()):
            output[index] = self._impl.decode_bit(float(probability))
        return output

