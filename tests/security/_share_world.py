"""Shared fixture world for the sharing-protocol tests (PUBLIC test keys only).

Hospitals A (sender), B and C (recipients); a root-signed registry; A's signed
manifest; identity stub pipelines so the protocol runs on CPU without a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.share.encoding import PROFILE_M1_PRIVATE_RANGE_V2 as PROFILE
from src.share.keys import TestFixtureStream, fixture_x25519
from src.share.manifest import PROFILE_MANIFESTS, sign_manifest
from src.share.receiver import RecipientIdentity
from src.share.registry import KeyRecord, TrustStore, build_snapshot
from src.share.sender import SenderIdentity


NOW = 1_800_000_000
MANIFEST = PROFILE_MANIFESTS[PROFILE]


def ed25519(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(TestFixtureStream("ed25519/" + label).read(32))


def identity(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu", torch.float32).clone()


@dataclass
class FactorySpy:
    calls: int = 0
    invert_calls: int = 0
    generate_calls: int = 0

    def __call__(self, manifest):
        self.calls += 1

        def invert(x):
            self.invert_calls += 1
            return identity(x)

        def generate(x):
            self.generate_calls += 1
            return identity(x)
        return invert, generate


@dataclass
class World:
    tmp_path: object
    b_status: str = "active"
    root: Ed25519PrivateKey = field(default_factory=lambda: ed25519("root"))
    a_sign: Ed25519PrivateKey = field(default_factory=lambda: ed25519("A"))

    def __post_init__(self) -> None:
        self.sender = SenderIdentity("hospital-A", "A-ed25519-2026a", self.a_sign)
        self.b = RecipientIdentity("hospital-B", "B-x25519-2026a", fixture_x25519("B"))
        self.c = RecipientIdentity("hospital-C", "C-x25519-2026a", fixture_x25519("C"))
        records = [
            KeyRecord("hospital-A", "A-ed25519-2026a", "ed25519-sign", "active", self.a_sign.public_key().public_bytes_raw()),
            KeyRecord("hospital-B", "B-x25519-2026a", "x25519-recv", self.b_status, self.b.private_key.public_key().public_bytes_raw()),
            KeyRecord("hospital-C", "C-x25519-2026a", "x25519-recv", "active", self.c.private_key.public_key().public_bytes_raw()),
        ]
        snapshot = build_snapshot(self.root, 1, NOW - 10, NOW + 3600, records)
        self.now = NOW                                    # test clock; advance it to exercise use-time checks
        self.registry = TrustStore(self.root.public_key(), self.tmp_path / "trust.json", clock=lambda: self.now)
        self.registry.load(snapshot)
        self.manifest_blobs = {MANIFEST.digest(): sign_manifest(MANIFEST, self.a_sign)}

    def fetch_manifest(self, digest: bytes) -> bytes:
        return self.manifest_blobs[digest]


def x0_fixture(seed: int = 1911) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    base = np.linspace(-0.9, 0.9, 65536, dtype=np.float32).reshape(256, 256)
    return torch.from_numpy(base + rng.normal(0, 0.1, (256, 256)).astype(np.float32)).reshape(1, 1, 256, 256)
