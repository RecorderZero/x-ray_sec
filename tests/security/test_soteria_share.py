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

import numpy as np
import pytest
import torch
from PIL import Image

import src.share.receiver as receiver_mod
import src.share.sender as sender_mod
from src.share.container import Header, parse
from src.share.keys import SecretKey, TestFixtureStream
from src.share.receiver import recover_image
from src.share.sender import share_image
from tests.security import _share_reference as ref
from tests.security._share_world import MANIFEST, FactorySpy, World, identity, x0_fixture


FIXED_NORM = 248.261


def _share(world: World, label: str | None, x0=None, invert=identity, generate=identity) -> bytes:
    return share_image(x0 if x0 is not None else x0_fixture(), sender=world.sender, recipient_inst_id=world.b.inst_id,
                       recipient_key_id=world.b.key_id, registry=world.registry, manifest_digest=MANIFEST.digest(),
                       invert=invert, generate=generate, fixture=TestFixtureStream(label) if label else None)


def _recover(world: World, blob: bytes, factory=None):
    return recover_image(blob, recipient=world.b, registry=world.registry, fetch_manifest=world.fetch_manifest,
                         pipeline_factory=factory or FactorySpy())


# ===================================================================== P3 independent reference (AF-054)
def p3_check(world: World, x0: torch.Tensor, label: str) -> dict:
    """Raise AssertionError unless the service agrees with the from-x0 reference.

    (a) metadata: the service's norm/lo/hi (opened with an independent key) must
        equal the reference's own values computed from x0 -- catches sender
        metadata faults such as norm x 1.01;
    (b) encode half: max|dq| <= 1 and <= 1e-3 of pixels differ (POST-HOC criterion,
        see module notes; validated on held-out seeds by the P3 report script);
    (c) decode half: the service output equals the reference decode of the
        service's public q with the REFERENCE's own norm/lo/hi, max-abs < 1e-5
        (the pre-registered P3 gate).
    Returns the lossy error vs x0 for reporting (no quality gate).
    """
    blob = _share(world, label, x0=x0)
    expected = ref.encode(x0.numpy(), label)
    norm_s, lo_s, hi_s = ref.service_metadata(blob, label)
    assert abs(norm_s - expected["norm"]) <= 1e-9 * expected["norm"], "sender norm disagrees with reference"
    assert abs(float(lo_s) - float(expected["lo"])) <= 1e-5 and abs(float(hi_s) - float(expected["hi"])) <= 1e-5, \
        "sender lo/hi disagree with reference"
    q_s = ref.service_q(blob)
    dq = np.abs(q_s.astype(np.int16) - expected["q"].astype(np.int16))
    assert int(dq.max()) <= 1 and float((dq > 0).mean()) <= 1e-3, "encode half disagrees beyond rint-boundary flips"
    out = _recover(world, blob).numpy().reshape(256, 256)
    oracle = ref.decode(q_s, expected["norm"], expected["lo"], expected["hi"], label)
    assert float(np.abs(out - oracle).max()) < 1e-5, "service output disagrees with reference decode"
    diff = out.astype(np.float64) - x0.numpy().reshape(256, 256).astype(np.float64)
    return {"dq_pixels": int((dq > 0).sum()), "dq_max": int(dq.max()),
            "lossy_max_abs": float(np.abs(diff).max()), "lossy_rmse": float(np.sqrt((diff ** 2).mean()))}


@pytest.mark.parametrize("seed", ref.THRESHOLD_SETTING_SEEDS[:5])
def test_p3_service_matches_independent_reference(tmp_path, seed: int) -> None:
    stats = p3_check(World(tmp_path), x0_fixture(seed), f"P3/{seed}")
    assert stats["lossy_max_abs"] > 0          # PNG path is genuinely lossy; values are reported, not gated


def test_p3_quantizer_is_bit_identical_on_identical_float32_input() -> None:
    """Isolate rounding from FWHT precision: same float32 input -> same q, incl. exact .5 boundaries."""
    from src.share.container import encode_anon_png
    rng = np.random.default_rng(1911)
    for trial in range(4):
        x = rng.normal(0, 1, (256, 256)).astype(np.float32)
        lo, hi = np.float32(-3), np.float32(5)
        x[0, 0], x[0, 1] = lo, hi
        # place values exactly on k + 0.5 quantization boundaries
        x[1, :10] = lo + (np.arange(10, dtype=np.float32) + np.float32(0.5)) / np.float32(255) * np.float32(hi - lo)
        x = np.clip(x, lo, hi)
        png, s_lo, s_hi = encode_anon_png(x)
        q_ref, r_lo, r_hi = ref.quantize(x)
        with Image.open(io.BytesIO(png)) as im:
            assert np.array_equal(np.asarray(im, dtype=np.uint8), q_ref)
        assert (s_lo, s_hi) == (r_lo, r_hi)
    png, s_lo, s_hi = encode_anon_png(np.full((256, 256), 0.5, dtype=np.float32))
    q_ref, *_ = ref.quantize(np.full((256, 256), 0.5, dtype=np.float32))
    with Image.open(io.BytesIO(png)) as im:
        assert np.array_equal(np.asarray(im, dtype=np.uint8), q_ref) and not q_ref.any()


@pytest.mark.parametrize("mutation", ["norm_x1.01", "lo_shift", "hi_shift", "lo_hi_swap_scale"])
def test_p3_catches_sender_metadata_faults(tmp_path, monkeypatch, mutation: str) -> None:
    """AF-054 acceptance: faults in what the sender writes must fail P3."""
    real = sender_mod.seal_metadata

    def faulty(key, h, norm, lo, hi):
        span = float(hi) - float(lo)
        if mutation == "norm_x1.01":
            norm = norm * 1.01
        elif mutation == "lo_shift":
            lo = float(lo) - 0.01 * span
        elif mutation == "hi_shift":
            hi = float(hi) + 0.01 * span
        else:
            lo, hi = float(lo) * 1.001, float(hi) * 1.001
        return real(key, h, norm, lo, hi)
    monkeypatch.setattr(sender_mod, "seal_metadata", faulty)
    with pytest.raises(AssertionError):
        p3_check(World(tmp_path), x0_fixture(1911), f"P3/mut/{mutation}")


# ===================================================================== V7 leak monitoring
def _secret_patterns(label: str) -> list[bytes]:
    stream = TestFixtureStream(label)
    k, nonce = stream.read(32), stream.read(16)
    secrets = [k] + [ref.subkey(k, nonce, lab) for lab in
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
