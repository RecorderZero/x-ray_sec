"""A-side encryption service (§8.3, A steps 1-8).

``invert`` / ``generate`` are injected: the real deployment passes the
DPM-Solver++(2M) n=100 null-backward pipeline, tests pass stubs. The image key
and all subkeys are wiped in ``finally`` on every exit path; nothing here logs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .container import Header, assemble, compute_tag, encode_anon_png, seal_metadata, sign
from .encoding import PROFILE_M1_PRIVATE_RANGE_V2, require_id, require_profile
from .errors import ShareFormatError
from .keys import TestFixtureStream, build_info, derive_subkeys, new_image_key, new_nonce, wipe_all, wrap_image_key
from .registry import TrustStore
from .transform import LATENT_SHAPE, encrypt_latent


Latent = torch.Tensor
Pipeline = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SenderIdentity:
    inst_id: str
    signing_key_id: str
    signing_key: Ed25519PrivateKey

    def __repr__(self) -> str:
        return f"SenderIdentity(inst_id={self.inst_id!r}, signing_key_id={self.signing_key_id!r}, signing_key=<redacted>)"


def _as_image(tensor: torch.Tensor) -> np.ndarray:
    if tuple(tensor.shape) != LATENT_SHAPE:
        raise ShareFormatError(f"generator output must have shape {LATENT_SHAPE}")
    array = tensor.detach().to("cpu", torch.float32).numpy().reshape(256, 256)
    if not np.isfinite(array).all():
        raise ShareFormatError("generator output must be finite")
    return array


def share_image(
    x0: torch.Tensor,
    *,
    sender: SenderIdentity,
    recipient_inst_id: str,
    recipient_key_id: str,
    registry: TrustStore,
    manifest_digest: bytes,
    invert: Pipeline,
    generate: Pipeline,
    profile_id: str = PROFILE_M1_PRIVATE_RANGE_V2,
    fixture: TestFixtureStream | None = None,
) -> bytes:
    """Return one signed container addressed to (recipient_inst_id, recipient_key_id).

    ``fixture`` is for reproducible research runs only: it replaces the CSPRNG
    image key/nonce with a PUBLIC labelled stream (§8.8 key policy).
    """
    require_profile(profile_id)
    if not isinstance(registry, TrustStore):
        raise ShareFormatError("registry must be a TrustStore (use-time freshness checks, AF-052)")
    require_id(recipient_inst_id, "inst_id_B")
    require_id(recipient_key_id, "key_id_B")
    recipient_pk = registry.recipient_key(recipient_inst_id, recipient_key_id)   # fail closed (RegistryError)
    image_key = subkeys = None
    try:
        image_key = fixture.image_key() if fixture is not None else new_image_key()
        nonce = fixture.nonce() if fixture is not None else new_nonce()
        info = build_info(profile_id, nonce, recipient_inst_id, recipient_key_id, sender.inst_id, sender.signing_key_id)
        wrap = wrap_image_key(image_key, recipient_pk, info)
        subkeys = derive_subkeys(image_key, nonce, profile_id)

        z = invert(x0)
        protected, original_norm = encrypt_latent(z, subkeys)
        png, lo, hi = encode_anon_png(_as_image(generate(protected)))

        h_bytes = Header(profile_id, manifest_digest, sender.inst_id, sender.signing_key_id,
                         recipient_inst_id, recipient_key_id, nonce, wrap).encode()
        meta_ct = seal_metadata(subkeys["private-metadata-aead"], h_bytes, original_norm, lo, hi)
        tag = compute_tag(subkeys["mac"], h_bytes, meta_ct, png)
        sig = sign(sender.signing_key, h_bytes, meta_ct, png, tag)
        return assemble(h_bytes, meta_ct, png, tag, sig)
    finally:
        wipe_all(image_key, subkeys)
