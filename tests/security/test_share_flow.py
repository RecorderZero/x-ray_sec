"""S7: end-to-end A -> B flow with stub pipelines (V3', V4 integration, V5, V8, spy)."""

import pytest
import torch

from src.crypto.errors import AuthenticationError
from src.share.container import assemble, compute_tag, parse, sign
from src.share.errors import ManifestError, RecipientError, RegistryError, SignatureError, UnwrapError
from src.share.keys import TestFixtureStream
from src.share.receiver import ReceiverService, recover_image
from src.share.sender import share_image

from tests.security._share_world import MANIFEST, FactorySpy, World, ed25519, identity, x0_fixture


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
