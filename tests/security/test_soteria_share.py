"""S8: remaining §8.8 checks -- P3 independent reference, V7 leak monitoring, V9 public view.

V1 lives in test_prng_v1_frozen.py, P1 in test_share_transform.py, P2/V4/V11 in
test_share_container.py, V3/V6 in test_share_keys.py, V5/V8 in test_share_registry.py
and test_share_flow.py. These are functional/correctness checks under stated
conditions, not general security proofs.
"""

from __future__ import annotations

import base64
import io
import logging
import struct
import traceback
from dataclasses import replace

import numpy as np
import pytest
import torch
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from PIL import Image

import src.share.receiver as receiver_mod
import src.share.sender as sender_mod
from src.crypto.prng import inverse_permutation, permutation_from_key, sign_vector_from_key
from src.share.container import Header, parse
from src.share.encoding import PROFILE_M1_PRIVATE_RANGE_V2 as PROFILE
from src.share.keys import SecretKey, TestFixtureStream
from src.share.receiver import recover_image
from src.share.sender import share_image
from tests.security._share_world import MANIFEST, FactorySpy, World, identity, x0_fixture


FIXED_NORM = 248.261


def _share(world: World, label: str | None, x0=None, invert=identity, generate=identity) -> bytes:
    return share_image(x0 if x0 is not None else x0_fixture(), sender=world.sender, recipient_inst_id=world.b.inst_id,
                       recipient_key_id=world.b.key_id, registry=world.registry, manifest_digest=MANIFEST.digest(),
                       invert=invert, generate=generate, fixture=TestFixtureStream(label) if label else None)


def _recover(world: World, blob: bytes, factory=None):
    return recover_image(blob, recipient=world.b, registry=world.registry, fetch_manifest=world.fetch_manifest,
                         pipeline_factory=factory or FactorySpy())


# ===================================================================== P3 independent reference
def _lp(b: bytes) -> bytes:
    return len(b).to_bytes(4, "big") + b


def _ref_subkey(k: bytes, nonce: bytes, label: str) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=nonce,
                info=_lp(PROFILE.encode()) + _lp(label.encode())).derive(k)


def _ref_fwht(v: np.ndarray) -> np.ndarray:
    out = v.astype(np.float64).copy()
    h = 1
    while h < out.size:
        out = out.reshape(-1, 2, h)
        a, b = out[:, 0, :].copy(), out[:, 1, :].copy()
        out[:, 0, :], out[:, 1, :] = a + b, a - b
        out = out.reshape(-1)
        h *= 2
    return out / np.sqrt(out.size)


def _ref_streams(label: str):
    stream = TestFixtureStream(label)
    k, nonce = stream.read(32), stream.read(16)
    pre = sign_vector_from_key(_ref_subkey(k, nonce, "round/0/sign-pre"), 65536).astype(np.float64)
    post = sign_vector_from_key(_ref_subkey(k, nonce, "round/0/sign-post"), 65536).astype(np.float64)
    perm = permutation_from_key(_ref_subkey(k, nonce, "round/0/perm"), 65536)
    return k, nonce, pre, perm, post


def _reference_encode(x0: np.ndarray, label: str) -> np.ndarray:
    """Spec-level encode half (float64 transform, float32 quantization as §8.4.1 (8)); returns q."""
    _, _, pre, perm, post = _ref_streams(label)
    z = x0.reshape(-1).astype(np.float64)
    c = (_ref_fwht(z * (FIXED_NORM / float(np.linalg.norm(z))) * pre)[perm] * post).astype(np.float32).reshape(256, 256)
    lo, hi = np.float32(c.min()), np.float32(c.max())
    return np.clip(np.rint((c - lo) / np.float32(hi - lo) * np.float32(255)), 0, 255).astype(np.uint8)


def _reference_decode(blob: bytes, label: str) -> np.ndarray:
    """Spec-level decode half fed with the service's public q; metadata opened with an independently derived key."""
    k, nonce, pre, perm, post = _ref_streams(label)
    p = parse(blob)
    meta_key = _ref_subkey(k, nonce, "private-metadata-aead")
    pt = ChaCha20Poly1305(meta_key).decrypt(bytes(12), p.meta_ct, b"SOTERIA\x00\x00\x02" + _lp(p.h_bytes))
    norm = struct.unpack("<d", pt[:8])[0]
    lo, hi = np.frombuffer(pt[8:12], "<f4")[0], np.frombuffer(pt[12:16], "<f4")[0]
    with Image.open(io.BytesIO(p.png)) as im:
        q = np.asarray(im, dtype=np.uint8)
    deq = (q.astype(np.float32) / np.float32(255) * np.float32(hi - lo) + lo).reshape(-1).astype(np.float64)
    u = _ref_fwht((deq * post)[inverse_permutation(perm)]) * pre
    return (u * (norm / FIXED_NORM)).astype(np.float32).reshape(256, 256)


@pytest.mark.parametrize("seed", [1911, 7, 42, 3, 5])
def test_p3_service_matches_independent_reference(tmp_path, seed: int) -> None:
    """P3 (§8.8). Decode half keeps the pre-registered max-abs < 1e-5 gate.

    Encode half: two independent float paths (torch float32 vs numpy float64 FWHT) can
    land a value on opposite sides of a rint .5 boundary, flipping q by exactly 1 at a
    handful of pixels (observed <= 1/65536 on 8 fixtures, 2026-11-04). The encode
    check is therefore max |dq| <= 1 with <= 1e-3 of pixels differing -- added after
    that observation and recorded as such in PROGRESS; a wrong rounding mode or scale
    would differ on ~half the pixels.
    """
    world = World(tmp_path)
    x0 = x0_fixture(seed)
    label = f"P3/{seed}"
    blob = _share(world, label, x0=x0)
    with Image.open(io.BytesIO(parse(blob).png)) as im:
        q_service = np.asarray(im, dtype=np.int16)
    dq = np.abs(q_service - _reference_encode(x0.numpy(), label).astype(np.int16))
    assert int(dq.max()) <= 1 and float((dq > 0).mean()) <= 1e-3
    out = _recover(world, blob).numpy().reshape(256, 256)
    assert float(np.abs(out - _reference_decode(blob, label)).max()) < 1e-5    # pre-registered P3 tolerance
    assert float(np.abs(out - x0.numpy().reshape(256, 256)).max()) > 0          # PNG path is genuinely lossy (reported)


# ===================================================================== V7 leak monitoring
def _secret_patterns(label: str) -> list[bytes]:
    stream = TestFixtureStream(label)
    k, nonce = stream.read(32), stream.read(16)
    secrets = [k] + [_ref_subkey(k, nonce, lab) for lab in
                     ("round/0/sign-pre", "round/0/perm", "round/0/sign-post", "mac", "private-metadata-aead")]
    pats = []
    for s in secrets:
        pats += [s, s.hex().encode(), s.hex().upper().encode(), base64.b64encode(s), base64.urlsafe_b64encode(s),
                 base64.b64encode(s).rstrip(b"=")]
    return pats


class _Boom(RuntimeError):
    pass


def _raiser(*_a, **_k):
    raise _Boom("injected failure")


SENDER_FAULTS = {
    "none": None,
    "hpke_wrap": ("wrap_image_key", "after"),
    "png": ("encode_anon_png", "replace"),
    "signature": ("sign", "replace"),
    "inference": ("invert", None),
}
RECEIVER_FAULTS = {
    "none": None,
    "pipeline_factory": "factory",
    "png_decode": ("decode_anon_png", "replace"),
    "generate": "generate",
}


def _scan(blobs: list[bytes], pats: list[bytes]) -> list[bytes]:
    return [p for p in pats for b in blobs if p and p in b]


@pytest.mark.parametrize("fault", list(SENDER_FAULTS))
def test_v7_sender_never_emits_key_material(tmp_path, monkeypatch, caplog, capsys, fault: str) -> None:
    world = World(tmp_path)
    label = f"V7/sender/{fault}"
    monkeypatch.chdir(tmp_path)
    wiped: list[SecretKey] = []
    real_wipe = sender_mod.wipe_all

    def recording_wipe(*keys):
        for item in keys:
            if item is not None:
                wiped.extend(item.values() if isinstance(item, dict) else [item])
        real_wipe(*keys)
    monkeypatch.setattr(sender_mod, "wipe_all", recording_wipe)
    invert = identity
    spec = SENDER_FAULTS[fault]
    if spec == ("invert", None):
        invert = _raiser
    elif spec:
        name, mode = spec
        real = getattr(sender_mod, name)
        if mode == "after":
            monkeypatch.setattr(sender_mod, name, lambda *a, **k: (real(*a, **k), _raiser())[1])
        else:
            monkeypatch.setattr(sender_mod, name, _raiser)
    caplog.set_level(logging.DEBUG)
    error_text = b""
    blob = b""
    try:
        blob = _share(world, label, invert=invert)
    except _Boom:
        error_text = "".join(traceback.format_exc()).encode()
    out = capsys.readouterr()
    files = [p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()]
    corpus = [error_text, out.out.encode(), out.err.encode(), caplog.text.encode(), *files]
    pats = _secret_patterns(label)
    assert bool(error_text) == (fault != "none"), "the injected fault must actually fire"
    assert not _scan(corpus, pats), "key material reached a monitored output"
    if fault == "none":
        assert not _scan([blob], pats[:6])            # the raw image key itself never appears in the container
    assert wiped and all(k.wiped and k.buffer_is_zero() for k in wiped)


@pytest.mark.parametrize("fault", list(RECEIVER_FAULTS))
def test_v7_receiver_never_emits_key_material(tmp_path, monkeypatch, caplog, capsys, fault: str) -> None:
    world = World(tmp_path)
    label = f"V7/receiver/{fault}"
    blob = _share(world, label)
    monkeypatch.chdir(tmp_path)
    wiped: list[SecretKey] = []
    real_wipe = receiver_mod.wipe_all

    def recording_wipe(*keys):
        for item in keys:
            if item is not None:
                wiped.extend(item.values() if isinstance(item, dict) else [item])
        real_wipe(*keys)
    monkeypatch.setattr(receiver_mod, "wipe_all", recording_wipe)
    spec = RECEIVER_FAULTS[fault]
    factory = FactorySpy()
    if spec == "factory":
        factory = _raiser
    elif spec == "generate":
        factory = lambda manifest: (identity, _raiser)
    elif spec:
        monkeypatch.setattr(receiver_mod, spec[0], _raiser)
    caplog.set_level(logging.DEBUG)
    error_text = b""
    try:
        _recover(world, blob, factory)
    except _Boom:
        error_text = "".join(traceback.format_exc()).encode()
    out = capsys.readouterr()
    files = [p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()]
    corpus = [error_text, out.out.encode(), out.err.encode(), caplog.text.encode(), *files]
    assert bool(error_text) == (fault != "none"), "the injected fault must actually fire"
    assert not _scan(corpus, _secret_patterns(label))
    assert wiped and all(k.wiped and k.buffer_is_zero() for k in wiped)


def test_v7_public_api_exposes_no_key_returning_function() -> None:
    for module in (sender_mod, receiver_mod):
        public = [n for n in dir(module) if not n.startswith("_")]
        for name in public:
            assert "unwrap" not in name.lower() or name == "unwrap_image_key"   # internal helper import only
    service_methods = [m for m in dir(receiver_mod.ReceiverService) if not m.startswith("_")]
    assert service_methods == ["recover"]


# ===================================================================== V9 public view
def test_v9_public_container_view(tmp_path) -> None:
    world = World(tmp_path)
    x0 = x0_fixture(3)
    blob = _share(world, "V9/a", x0=x0)
    p = parse(blob)                                               # strict schema: exactly the allowlisted fields
    assert set(Header.__dataclass_fields__) == {"profile_id", "manifest_digest", "inst_id_a", "signing_key_id_a",
                                                "inst_id_b", "key_id_b", "nonce", "wrap"}
    for chunk in (b"tEXt", b"iTXt", b"zTXt", b"x-range"):
        assert chunk not in p.png
    # plaintext norm / lo / hi never appear in the public bytes (any endianness / width)
    norm = float(np.linalg.norm(x0.numpy().astype(np.float64)))
    candidates = [struct.pack("<d", norm), struct.pack(">d", norm), struct.pack("<f", norm), struct.pack(">f", norm)]
    assert not any(c in blob for c in candidates)


def test_v9_same_patient_headers_repeat_only_allowed_fields(tmp_path) -> None:
    world = World(tmp_path)
    h1 = parse(_share(world, None, x0=x0_fixture(11))).header
    h2 = parse(_share(world, None, x0=x0_fixture(12))).header
    allowed_repeat = {"profile_id", "manifest_digest", "inst_id_a", "signing_key_id_a", "inst_id_b", "key_id_b"}
    for name in Header.__dataclass_fields__:
        if getattr(h1, name) == getattr(h2, name):
            assert name in allowed_repeat, name
    assert h1.nonce != h2.nonce and h1.wrap != h2.wrap


def test_repeated_recovery_is_allowed_and_documented_as_non_goal(tmp_path) -> None:
    """§8.2: no replay / duplicate-import protection; recovering twice is permitted."""
    world = World(tmp_path)
    blob = _share(world, "replay")
    assert torch.equal(_recover(world, blob), _recover(world, blob))
