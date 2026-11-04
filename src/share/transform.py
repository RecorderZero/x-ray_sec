"""v2 latent transform keyed by per-recipient subkeys (§8.3 step 5 / §8.4.1 (9)).

Same mathematics as the deployed v1 S2fn path -- fixed-norm, then one round of
``D_post P (H/sqrt d) D_pre`` -- but the sign/permutation streams come from the
subkeys derived from ``k_img_B`` instead of a master key. FWHT, fixed-norm
helpers and permutation inversion are reused from :mod:`src.crypto`.
"""

from __future__ import annotations

import numpy as np
import torch

from src.crypto.prng import inverse_permutation, permutation_from_key, sign_vector_from_key
from src.crypto.s2_fixed_norm import FIXED_NORM, normalize_to_fixed_norm, restore_original_norm
from src.crypto.sot import fwht

from .errors import ShareFormatError
from .keys import SecretKey


LATENT_SHAPE = (1, 1, 256, 256)
LATENT_DIM = 256 * 256


def _check_latent(latent: torch.Tensor) -> None:
    if not isinstance(latent, torch.Tensor) or tuple(latent.shape) != LATENT_SHAPE:
        raise ShareFormatError(f"latent must have shape {LATENT_SHAPE}")
    if latent.dtype != torch.float32:
        raise ShareFormatError("latent must be float32")
    if not bool(torch.isfinite(latent).all()):
        raise ShareFormatError("latent must be finite")


def _round_material(subkeys: dict[str, SecretKey], device: torch.device, dtype: torch.dtype):
    pre = torch.as_tensor(sign_vector_from_key(subkeys["round/0/sign-pre"].raw(), LATENT_DIM), device=device, dtype=dtype)
    post = torch.as_tensor(sign_vector_from_key(subkeys["round/0/sign-post"].raw(), LATENT_DIM), device=device, dtype=dtype)
    perm = permutation_from_key(subkeys["round/0/perm"].raw(), LATENT_DIM)
    return pre, perm, post


def dphd(u: torch.Tensor, subkeys: dict[str, SecretKey]) -> torch.Tensor:
    _check_latent(u)
    flat = u.reshape(1, -1)
    pre, perm, post = _round_material(subkeys, flat.device, flat.dtype)
    out = fwht(flat * pre).index_select(-1, torch.as_tensor(perm, device=flat.device, dtype=torch.long)) * post
    return out.reshape(LATENT_SHAPE)


def dphd_inverse(c: torch.Tensor, subkeys: dict[str, SecretKey]) -> torch.Tensor:
    _check_latent(c)
    flat = c.reshape(1, -1)
    pre, perm, post = _round_material(subkeys, flat.device, flat.dtype)
    inverse = torch.as_tensor(inverse_permutation(perm), device=flat.device, dtype=torch.long)
    out = fwht((flat * post).index_select(-1, inverse)) * pre
    return out.reshape(LATENT_SHAPE)


def encrypt_latent(z: torch.Tensor, subkeys: dict[str, SecretKey]) -> tuple[torch.Tensor, float]:
    """fixed-norm (C = FIXED_NORM) -> DPHD; returns the protected latent and ||z|| (float64)."""
    _check_latent(z)
    u, original_norm = normalize_to_fixed_norm(z)
    return dphd(u, subkeys), original_norm


def decrypt_latent(c: torch.Tensor, subkeys: dict[str, SecretKey], original_norm: float) -> torch.Tensor:
    """DPHD^-1 -> multiply by original_norm / C."""
    if not np.isfinite(original_norm) or original_norm <= 0:
        raise ShareFormatError("original norm must be positive and finite")
    return restore_original_norm(dphd_inverse(c, subkeys), original_norm)


__all__ = ["FIXED_NORM", "LATENT_SHAPE", "decrypt_latent", "dphd", "dphd_inverse", "encrypt_latent"]
