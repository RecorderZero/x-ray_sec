"""B-side decryption service (§8.3, B steps 1-8).

Order is enforced: bounded parse -> registry-pinned sender key -> signature ->
recipient check -> signed manifest -> HPKE unwrap -> MAC -> metadata AEAD.
Only after all of these pass is ``pipeline_factory(manifest)`` called, so an
unverified container never loads a model or touches the GPU. The image key and
subkeys are wiped in ``finally``; the public :class:`ReceiverService` returns only
the recovered image and reports every rejection as one generic error type.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from src.crypto.errors import AuthenticationError

from .container import decode_anon_png, open_metadata, parse, verify_signature, verify_tag
from .errors import RecipientError, ShareFormatError
from .keys import build_info, derive_subkeys, unwrap_image_key, wipe_all
from .manifest import Manifest, verify_manifest
from .registry import TrustStore
from .transform import LATENT_SHAPE, decrypt_latent


Pipeline = Callable[[torch.Tensor], torch.Tensor]
PipelineFactory = Callable[[Manifest], tuple[Pipeline, Pipeline]]


@dataclass(frozen=True)
class RecipientIdentity:
    inst_id: str
    key_id: str
    private_key: X25519PrivateKey

    def __repr__(self) -> str:
        return f"RecipientIdentity(inst_id={self.inst_id!r}, key_id={self.key_id!r}, private_key=<redacted>)"


def recover_image(
    container: bytes,
    *,
    recipient: RecipientIdentity,
    registry: TrustStore,
    fetch_manifest: Callable[[bytes], bytes],
    pipeline_factory: PipelineFactory,
) -> torch.Tensor:
    """Internal flow; raises the layer-specific error that fired (for tests / audit)."""
    if not isinstance(registry, TrustStore):
        raise ShareFormatError("registry must be a TrustStore (use-time freshness checks, AF-052)")
    parsed = parse(container)                                                      # 1
    header = parsed.header
    sender_pub = registry.signing_key(header.inst_id_a, header.signing_key_id_a)   # 2
    verify_signature(sender_pub, parsed.h_bytes, parsed.meta_ct, parsed.png, parsed.tag, parsed.sig)
    if header.inst_id_b != recipient.inst_id or header.key_id_b != recipient.key_id:   # 3
        raise RecipientError("container is not addressed to this recipient")
    manifest = verify_manifest(fetch_manifest(header.manifest_digest), header.manifest_digest,
                               sender_pub, header.profile_id)                      # 4
    info = build_info(header.profile_id, header.nonce, header.inst_id_b, header.key_id_b,
                      header.inst_id_a, header.signing_key_id_a)
    image_key = subkeys = None
    try:
        image_key = unwrap_image_key(header.wrap, recipient.private_key, info)     # 5
        subkeys = derive_subkeys(image_key, header.nonce, header.profile_id)
        verify_tag(subkeys["mac"], parsed.h_bytes, parsed.meta_ct, parsed.png, parsed.tag)
        original_norm, lo, hi = open_metadata(subkeys["private-metadata-aead"], parsed.h_bytes, parsed.meta_ct)  # 6
        anon = decode_anon_png(parsed.png, lo, hi)                                 # 7
        invert, generate = pipeline_factory(manifest)
        protected_hat = invert(torch.from_numpy(anon).reshape(LATENT_SHAPE))
        if not isinstance(protected_hat, torch.Tensor) or tuple(protected_hat.shape) != LATENT_SHAPE:
            raise ShareFormatError("inverter output has the wrong shape")
        z_hat = decrypt_latent(protected_hat.detach().to("cpu", torch.float32), subkeys, original_norm)
        return generate(z_hat)                                                     # 8
    finally:
        wipe_all(image_key, subkeys)


class ReceiverService:
    """Public service boundary: image out, or one generic rejection (no key, no layer detail)."""

    def __init__(self, recipient: RecipientIdentity, registry: TrustStore,
                 fetch_manifest: Callable[[bytes], bytes], pipeline_factory: PipelineFactory):
        self._recipient = recipient
        self._registry = registry
        self._fetch_manifest = fetch_manifest
        self._pipeline_factory = pipeline_factory

    def recover(self, container: bytes) -> torch.Tensor:
        try:
            return recover_image(container, recipient=self._recipient, registry=self._registry,
                                 fetch_manifest=self._fetch_manifest, pipeline_factory=self._pipeline_factory)
        except (AuthenticationError, ShareFormatError):
            raise AuthenticationError("share container rejected") from None

    def __repr__(self) -> str:
        return f"ReceiverService(recipient={self._recipient!r})"
