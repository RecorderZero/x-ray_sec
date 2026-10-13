"""D5.3 ChaCha20-backed signs and unbiased Fisher-Yates permutations.

HKDF output is used directly as a 256-bit ChaCha20 key. It is never converted
to an integer seed and never enters Python, NumPy, or Torch PRNGs.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from .errors import CryptoConfigurationError
from .kdf import derive_subkey, require_master_key, require_nonce, round_label


CHACHA20_IV = bytes(16)  # 32-bit counter=0 || fixed 96-bit stream nonce.


class ByteSource(Protocol):
    def read(self, size: int) -> bytes: ...


class ChaCha20Stream:
    """Stateful original-ChaCha20 stream with a zero initial counter.

    The per-image nonce is already consumed as the HKDF salt, so each
    (master key, image nonce, round, purpose) tuple has an independent stream
    key.  Keeping the cipher IV fixed prevents random image-nonce bytes from
    becoming a near-overflow 32-bit initial counter (AF-037).
    """

    def __init__(self, key: bytes):
        require_master_key(key)
        self._encryptor = Cipher(algorithms.ChaCha20(key, CHACHA20_IV), mode=None).encryptor()

    def read(self, size: int) -> bytes:
        if not isinstance(size, int) or size < 0:
            raise CryptoConfigurationError("stream read size must be non-negative")
        return self._encryptor.update(bytes(size))


def _uniform_below(source: ByteSource, bound: int) -> int:
    """Uniform integer in [0,bound), rejecting the incomplete uint32 tail."""
    if not isinstance(bound, int) or not (1 <= bound <= 2**32):
        raise CryptoConfigurationError("bound must be in 1..2^32")
    modulus = 1 << 32
    limit = modulus - modulus % bound
    while True:
        block = source.read(4)
        if len(block) != 4:
            raise RuntimeError("CSPRNG source returned a short block")
        value = int.from_bytes(block, "little")
        if value < limit:
            return value % bound


def sign_vector(master_key: bytes, nonce: bytes, round_index: int, purpose: str, size: int) -> np.ndarray:
    if purpose not in {"sign-pre", "sign-post"}:
        raise CryptoConfigurationError("sign purpose must be sign-pre or sign-post")
    if not isinstance(size, int) or size < 1:
        raise CryptoConfigurationError("sign vector size must be positive")
    key = derive_subkey(master_key, nonce, round_label(round_index, purpose))
    packed = ChaCha20Stream(key).read((size + 7) // 8)
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")[:size]
    return np.where(bits == 0, -1, 1).astype(np.int8)


def permutation(master_key: bytes, nonce: bytes, round_index: int, size: int) -> np.ndarray:
    if not isinstance(size, int) or size < 1 or size > 2**32:
        raise CryptoConfigurationError("permutation size must be in 1..2^32")
    key = derive_subkey(master_key, nonce, round_label(round_index, "perm"))
    source = ChaCha20Stream(key)
    result = np.arange(size, dtype=np.int64)
    for index in range(size - 1, 0, -1):
        other = _uniform_below(source, index + 1)
        result[index], result[other] = result[other], result[index]
    return result


def inverse_permutation(indices: np.ndarray) -> np.ndarray:
    array = np.asarray(indices)
    if array.ndim != 1 or not np.array_equal(np.sort(array), np.arange(len(array))):
        raise CryptoConfigurationError("input is not a permutation")
    inverse = np.empty_like(array)
    inverse[array] = np.arange(len(array), dtype=array.dtype)
    return inverse
