"""S7: end-to-end A -> B flow with stub pipelines (V3', V4 integration, V5, V8, spy)."""

import pytest
import torch

from src.crypto.errors import AuthenticationError
from src.share.container import assemble, compute_tag, parse, sign
from src.share.errors import ManifestError, RecipientError, RegistryError, SignatureError, UnwrapError
from src.share.keys import TestFixtureStream
from src.share.receiver import ReceiverService, recover_image
from src.share.sender import share_image

from tests.security._share_world import MANIFEST, NOW, FactorySpy, World, ed25519, identity, x0_fixture

NOW_PLUS_EXPIRED = NOW + 3600


def _share(world: World, to: str = "B", fixture: str | None = None, x0=None) -> bytes:
    rec = world.b if to == "B" else world.c
    return share_image(x0 if x0 is not None else x0_fixture(), sender=world.sender,
                       recipient_inst_id=rec.inst_id, recipient_key_id=rec.key_id, registry=world.registry,
                       manifest_digest=MANIFEST.digest(), invert=identity, generate=identity,
                       fixture=TestFixtureStream(fixture) if fixture else None)


def _recover(world: World, blob: bytes, who: str = "B", spy: FactorySpy | None = None):
    rec = world.b if who == "B" else world.c
    return recover_image(blob, recipient=rec, registry=world.registry, fetch_manifest=world.fetch_manifest,
                         pipeline_factory=spy or FactorySpy())


def test_b_recovers_with_stub_pipeline(tmp_path) -> None:
    world = World(tmp_path)
    spy = FactorySpy()
    out = _recover(world, _share(world), spy=spy)
    assert tuple(out.shape) == (1, 1, 256, 256) and spy.calls == 1
    err = float((out - x0_fixture()).abs().max())
    assert err < 0.05, err   # identity stubs: only 8-bit quantization of the protected latent remains (reported in S8)


def test_wrong_recipient_rejected_before_any_inference(tmp_path) -> None:
    world = World(tmp_path)
    blob = _share(world, "B")
    spy = FactorySpy()
    with pytest.raises(RecipientError):
        _recover(world, blob, "C", spy)
    assert spy.calls == 0


def test_readdressed_and_resigned_container_still_cannot_be_unwrapped_by_c(tmp_path) -> None:
    """V3': even if the header is re-addressed to C and re-signed by the test signer, HPKE fails."""
    world = World(tmp_path)
    p = parse(_share(world, "B"))
    from dataclasses import replace
    h2 = replace(p.header, inst_id_b=world.c.inst_id, key_id_b=world.c.key_id).encode()
    sig2 = sign(world.a_sign, h2, p.meta_ct, p.png, p.tag)
    spy = FactorySpy()
    with pytest.raises(UnwrapError):
        _recover(world, assemble(h2, p.meta_ct, p.png, p.tag, sig2), "C", spy)
    assert spy.calls == 0


def test_wrap_swap_between_objects_hits_hpke_info_binding(tmp_path) -> None:
    world = World(tmp_path)
    p1, p2 = parse(_share(world)), parse(_share(world))
    from dataclasses import replace
    h = replace(p1.header, wrap=p2.header.wrap).encode()
    sig = sign(world.a_sign, h, p1.meta_ct, p1.png, p1.tag)
    spy = FactorySpy()
    with pytest.raises(UnwrapError):
        _recover(world, assemble(h, p1.meta_ct, p1.png, p1.tag, sig), "B", spy)
    assert spy.calls == 0


def test_unknown_or_unregistered_signer_rejected(tmp_path) -> None:
    world = World(tmp_path)
    p = parse(_share(world))
    from dataclasses import replace
    h = replace(p.header, signing_key_id_a="A-ed25519-unknown").encode()
    sig = sign(world.a_sign, h, p.meta_ct, p.png, p.tag)
    with pytest.raises(RegistryError):
        _recover(world, assemble(h, p.meta_ct, p.png, p.tag, sig))
    forged = sign(ed25519("mallory"), p.h_bytes, p.meta_ct, p.png, p.tag)
    with pytest.raises(SignatureError):
        _recover(world, assemble(p.h_bytes, p.meta_ct, p.png, p.tag, forged))


def test_manifest_problems_stop_before_unwrap(tmp_path) -> None:
    world = World(tmp_path)
    blob = _share(world)
    world.manifest_blobs[MANIFEST.digest()] = world.manifest_blobs[MANIFEST.digest()][:-1] + b"\x00"
    spy = FactorySpy()
    with pytest.raises(ManifestError):
        _recover(world, blob, spy=spy)
    assert spy.calls == 0


def test_sender_refuses_revoked_or_unknown_recipient(tmp_path) -> None:
    world = World(tmp_path, b_status="revoked")
    with pytest.raises(RegistryError):
        _share(world, "B")
    with pytest.raises(RegistryError):
        share_image(x0_fixture(), sender=world.sender, recipient_inst_id="hospital-B", recipient_key_id="B-x25519-nope",
                    registry=world.registry, manifest_digest=MANIFEST.digest(), invert=identity, generate=identity)


def test_service_boundary_reports_one_generic_error(tmp_path) -> None:
    world = World(tmp_path)
    blob = _share(world, "B")
    service_c = ReceiverService(world.c, world.registry, world.fetch_manifest, FactorySpy())
    with pytest.raises(AuthenticationError) as info:
        service_c.recover(blob)
    assert type(info.value) is AuthenticationError and str(info.value) == "share container rejected"
    assert info.value.__cause__ is None and info.value.__suppress_context__
    service_b = ReceiverService(world.b, world.registry, world.fetch_manifest, FactorySpy())
    assert tuple(service_b.recover(blob).shape) == (1, 1, 256, 256)
    assert "private_key=<redacted>" in repr(service_b)


def test_fixture_mode_is_reproducible_and_csprng_mode_is_fresh(tmp_path) -> None:
    world = World(tmp_path)
    f1, f2 = parse(_share(world, fixture="E1/x")), parse(_share(world, fixture="E1/x"))
    assert f1.png == f2.png and f1.header.nonce == f2.header.nonce   # anon image reproducible
    assert f1.header.wrap != f2.header.wrap                           # HPKE ephemeral still random
    r1, r2 = parse(_share(world)), parse(_share(world))
    assert r1.header.nonce != r2.header.nonce and r1.png != r2.png


# ------------------------------------------------------------------ AF-052 regressions (services)
class _CountingInvert:
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return identity(x)


def test_af052_long_lived_services_reject_after_expiry_before_inference(tmp_path) -> None:
    world = World(tmp_path)
    blob = _share(world)
    spy = FactorySpy()
    service = ReceiverService(world.b, world.registry, world.fetch_manifest, spy)
    service.recover(blob)                                   # within validity
    world.now = NOW_PLUS_EXPIRED
    with pytest.raises(AuthenticationError):
        service.recover(blob)
    assert spy.calls == 1                                    # the rejected request never reached the model
    inv = _CountingInvert()
    with pytest.raises(RegistryError):
        share_image(x0_fixture(), sender=world.sender, recipient_inst_id=world.b.inst_id, recipient_key_id=world.b.key_id,
                    registry=world.registry, manifest_digest=MANIFEST.digest(), invert=inv, generate=identity)
    assert inv.calls == 0


def test_af052_cached_services_reject_after_newer_revocation_is_persisted(tmp_path) -> None:
    from src.share.registry import KeyRecord, TrustStore, build_snapshot
    world = World(tmp_path)
    blob = _share(world)
    spy = FactorySpy()
    service = ReceiverService(world.b, world.registry, world.fetch_manifest, spy)
    revoked = [KeyRecord(r.inst_id, r.key_id, r.purpose, "revoked" if r.inst_id in ("hospital-A", "hospital-B") else r.status,
                         r.public_key) for r in world.registry.records]
    other = TrustStore(world.root.public_key(), tmp_path / "trust.json", clock=lambda: world.now)   # same state path
    other.load(build_snapshot(world.root, 2, NOW - 10, NOW + 3600, revoked))
    with pytest.raises(AuthenticationError):
        service.recover(blob)                                # cached v1 store is now stale
    with pytest.raises(RegistryError):
        _share(world)                                        # cached v1 sender path refuses too
    assert spy.calls == 0
    with pytest.raises(RegistryError):
        other.recipient_key(world.b.inst_id, world.b.key_id)  # and v2 says revoked


def test_raw_registry_object_is_not_accepted_by_services(tmp_path) -> None:
    world = World(tmp_path)
    with pytest.raises(Exception):
        share_image(x0_fixture(), sender=world.sender, recipient_inst_id=world.b.inst_id, recipient_key_id=world.b.key_id,
                    registry=world.registry.current, manifest_digest=MANIFEST.digest(), invert=identity, generate=identity)


# ------------------------------------------------------------------ AF-053 partial-derivation cleanup
import src.share.keys as keys_mod  # noqa: E402
from src.share.keys import SecretKey  # noqa: E402


class _Interrupt(KeyboardInterrupt):
    pass


def _tracking(monkeypatch) -> list:
    created = []
    original = SecretKey.__init__

    def init(self, material):
        original(self, material)
        created.append(self)
    monkeypatch.setattr(SecretKey, "__init__", init)
    return created


def _failing_hkdf(monkeypatch, fail_at: int, exc: type[BaseException]) -> None:
    real = keys_mod.HKDF
    state = {"n": 0}

    class FailingHKDF:
        def __init__(self, *a, **k):
            self._inner = real(*a, **k)

        def derive(self, ikm):
            state["n"] += 1
            if state["n"] == fail_at:
                raise exc("injected HKDF failure")
            return self._inner.derive(ikm)
    monkeypatch.setattr(keys_mod, "HKDF", FailingHKDF)


@pytest.mark.parametrize("exc", [RuntimeError, _Interrupt])
@pytest.mark.parametrize("fail_at", [1, 2, 3, 4, 5])
def test_af053_sender_wipes_every_buffer_when_any_hkdf_step_fails(tmp_path, monkeypatch, fail_at, exc) -> None:
    world = World(tmp_path)
    created = _tracking(monkeypatch)
    _failing_hkdf(monkeypatch, fail_at, exc)
    with pytest.raises(exc):
        _share(world)
    assert len(created) == fail_at            # image key + (fail_at - 1) finished subkeys
    assert all(k.wiped and k.buffer_is_zero() for k in created)


@pytest.mark.parametrize("exc", [RuntimeError, _Interrupt])
@pytest.mark.parametrize("fail_at", [1, 2, 3, 4, 5])
def test_af053_receiver_wipes_every_buffer_when_any_hkdf_step_fails(tmp_path, monkeypatch, fail_at, exc) -> None:
    world = World(tmp_path)
    blob = _share(world)
    created = _tracking(monkeypatch)
    _failing_hkdf(monkeypatch, fail_at, exc)
    spy = FactorySpy()
    with pytest.raises(exc):
        _recover(world, blob, spy=spy)
    assert spy.calls == 0
    assert len(created) == fail_at            # unwrapped image key + finished subkeys
    assert all(k.wiped and k.buffer_is_zero() for k in created)


def test_af053_normal_paths_still_wipe_everything(tmp_path, monkeypatch) -> None:
    world = World(tmp_path)
    created = _tracking(monkeypatch)
    blob = _share(world)
    _recover(world, blob)
    assert len(created) == 12 and all(k.wiped and k.buffer_is_zero() for k in created)


# ------------------------------------------------------------------ AF-055 manifest fetch contract
@pytest.mark.parametrize("failure", [KeyError, LookupError, FileNotFoundError, ConnectionError])
def test_af055_unavailable_manifest_is_a_controlled_rejection(tmp_path, failure) -> None:
    world = World(tmp_path)
    blob = _share(world)

    def missing(digest):
        raise failure("not available")
    spy = FactorySpy()
    with pytest.raises(ManifestError):
        recover_image(blob, recipient=world.b, registry=world.registry, fetch_manifest=missing, pipeline_factory=spy)
    service = ReceiverService(world.b, world.registry, missing, spy)
    with pytest.raises(AuthenticationError) as info:
        service.recover(blob)
    assert type(info.value) is AuthenticationError and spy.calls == 0


def test_af055_programming_errors_and_cancellation_are_not_swallowed(tmp_path) -> None:
    world = World(tmp_path)
    blob = _share(world)
    for exc in (TypeError, KeyboardInterrupt):
        def broken(digest, exc=exc):
            raise exc("bug or cancel")
        with pytest.raises(exc):
            ReceiverService(world.b, world.registry, broken, FactorySpy()).recover(blob)
