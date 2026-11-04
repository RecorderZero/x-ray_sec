"""S5: signed registry, pinned root, persisted anti-rollback (V5, V8, opinion F1)."""

import json
import struct

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.share.errors import RegistryError
from src.share.keys import TestFixtureStream, fixture_x25519
from src.share.registry import KeyRecord, TrustStore, build_snapshot


NOW = 1_800_000_000


def _root(label: str = "root") -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(TestFixtureStream("registry/" + label).read(32))


def _records(b_status: str = "active") -> list[KeyRecord]:
    a_sign = Ed25519PrivateKey.from_private_bytes(TestFixtureStream("ed25519/A").read(32)).public_key()
    return [
        KeyRecord("hospital-A", "A-ed25519-2026a", "ed25519-sign", "active", a_sign.public_bytes_raw()),
        KeyRecord("hospital-B", "B-x25519-2026a", "x25519-recv", b_status, fixture_x25519("B").public_key().public_bytes_raw()),
        KeyRecord("hospital-C", "C-x25519-2026a", "x25519-recv", "active", fixture_x25519("C").public_key().public_bytes_raw()),
    ]


def _snap(version: int, *, b_status: str = "active", issued=NOW - 10, expires=NOW + 3600, root=None, records=None) -> bytes:
    return build_snapshot(root or _root(), version, issued, expires, records if records is not None else _records(b_status))


def _store(tmp_path, clock=lambda: NOW) -> TrustStore:
    return TrustStore(_root().public_key(), tmp_path / "trust_state.json", clock=clock)


def test_valid_snapshot_loads_and_lookups_work(tmp_path) -> None:
    store = _store(tmp_path)
    store.load(_snap(1), now=NOW)
    assert store.recipient_key("hospital-B", "B-x25519-2026a").public_bytes_raw() == fixture_x25519("B").public_key().public_bytes_raw()
    store.signing_key("hospital-A", "A-ed25519-2026a")
    assert json.loads((tmp_path / "trust_state.json").read_text())["highest_version"] == 1


@pytest.mark.parametrize("inst,key,purpose", [
    ("hospital-B", "B-x25519-unknown", "recv"),   # unknown key id
    ("hospital-Z", "B-x25519-2026a", "recv"),     # unknown institution
    ("hospital-A", "A-ed25519-2026a", "recv"),    # signing key used as recipient key
    ("hospital-B", "B-x25519-2026a", "sign"),     # recipient key used as signing key
])
def test_lookup_fails_closed(tmp_path, inst, key, purpose) -> None:
    store = _store(tmp_path)
    store.load(_snap(1), now=NOW)
    with pytest.raises(RegistryError):
        (store.recipient_key if purpose == "recv" else store.signing_key)(inst, key)


def test_revoked_key_is_refused(tmp_path) -> None:
    store = _store(tmp_path)
    store.load(_snap(2, b_status="revoked"), now=NOW)
    with pytest.raises(RegistryError):
        store.recipient_key("hospital-B", "B-x25519-2026a")


def _payload_offset(snapshot: bytes) -> int:
    return 4 + 4 + len(b"soteria-share/registry/v1")   # LP(payload) prefix + LP(domain)


@pytest.mark.parametrize("field_offset", [0, 8, 16])   # version, issued_at, expires_at
def test_editing_version_or_validity_breaks_signature(tmp_path, field_offset: int) -> None:
    snap = bytearray(_snap(1))
    pos = _payload_offset(bytes(snap)) + field_offset + 7
    snap[pos] ^= 0x01
    with pytest.raises(RegistryError):
        _store(tmp_path).load(bytes(snap), now=NOW)


def test_mixing_in_old_record_breaks_signature(tmp_path) -> None:
    from src.share.encoding import lp
    from src.share.registry import build_payload
    new = _snap(3, b_status="revoked")
    revoked_b = _records("revoked")[1].encode()
    active_b = _records("active")[1].encode()
    new_payload = build_payload(3, NOW - 10, NOW + 3600, _records("revoked"))
    assert revoked_b in new_payload
    spliced_payload = new_payload.replace(revoked_b, active_b)   # old "active" record, new version header
    spliced = lp(spliced_payload) + new[-64:]                    # keep the new snapshot's signature
    with pytest.raises(RegistryError):
        _store(tmp_path).load(spliced, now=NOW)


def test_replay_of_older_snapshot_is_rejected_even_after_restart(tmp_path) -> None:
    store = _store(tmp_path)
    store.load(_snap(5, b_status="revoked"), now=NOW)
    with pytest.raises(RegistryError):
        store.load(_snap(4, b_status="active"), now=NOW)
    restarted = _store(tmp_path)   # same persisted state file
    with pytest.raises(RegistryError):
        restarted.load(_snap(4, b_status="active"), now=NOW)


def test_same_version_different_content_is_rejected(tmp_path) -> None:
    store = _store(tmp_path)
    store.load(_snap(6, b_status="active"), now=NOW)
    store.load(_snap(6, b_status="active"), now=NOW)   # identical re-load is fine
    with pytest.raises(RegistryError):
        store.load(_snap(6, b_status="revoked"), now=NOW)


def test_unverified_high_version_does_not_poison_state(tmp_path) -> None:
    store = _store(tmp_path)
    store.load(_snap(2), now=NOW)
    forged = _snap(999, root=_root("attacker"))
    with pytest.raises(RegistryError):
        store.load(forged, now=NOW)
    assert json.loads((tmp_path / "trust_state.json").read_text())["highest_version"] == 2
    store.load(_snap(3), now=NOW)   # legitimate progress still possible


def test_expired_or_future_snapshot_rejected_without_state_change(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(RegistryError):
        store.load(_snap(1, issued=NOW - 100, expires=NOW - 1), now=NOW)
    with pytest.raises(RegistryError):
        store.load(_snap(1, issued=NOW + 100, expires=NOW + 200), now=NOW)
    assert not (tmp_path / "trust_state.json").exists()


def test_pinned_root_cannot_be_swapped_by_snapshot(tmp_path) -> None:
    with pytest.raises(RegistryError):
        _store(tmp_path).load(_snap(1, root=_root("other-root")), now=NOW)


@pytest.mark.parametrize("mutate", ["trailing", "truncated", "oversize", "dup_record"])
def test_malformed_snapshots(tmp_path, mutate: str) -> None:
    if mutate == "trailing":
        snap = _snap(1) + b"\x00"
    elif mutate == "truncated":
        snap = _snap(1)[:-10]
    elif mutate == "oversize":
        snap = bytes((1 << 20) + 1)
    else:
        rec = _records()
        payload_records = rec + [rec[1]]
        # build_snapshot sorts; duplicates must still be rejected on parse
        snap = build_snapshot(_root(), 1, NOW - 10, NOW + 3600, payload_records)
    with pytest.raises(RegistryError):
        _store(tmp_path).load(snap, now=NOW)


def test_corrupt_state_file_fails_closed(tmp_path) -> None:
    (tmp_path / "trust_state.json").write_text("{not json")
    with pytest.raises(RegistryError):
        _store(tmp_path).load(_snap(1), now=NOW)


# ------------------------------------------------------------------ AF-052 use-time freshness
def test_af052_cached_store_rejects_after_expiry(tmp_path) -> None:
    clock = {"now": NOW}
    store = _store(tmp_path, clock=lambda: clock["now"])
    store.load(_snap(1, expires=NOW + 100), now=NOW)
    store.recipient_key("hospital-B", "B-x25519-2026a")            # valid now
    clock["now"] = NOW + 100                                       # expires_at reached
    with pytest.raises(RegistryError):
        store.recipient_key("hospital-B", "B-x25519-2026a")
    with pytest.raises(RegistryError):
        store.signing_key("hospital-A", "A-ed25519-2026a")


def test_af052_store_superseded_by_another_store_on_same_state_is_stale(tmp_path) -> None:
    old_store = _store(tmp_path)
    old_store.load(_snap(1, b_status="active"), now=NOW)
    new_store = _store(tmp_path)                                    # e.g. another worker / restarted service
    new_store.load(_snap(2, b_status="revoked"), now=NOW)
    with pytest.raises(RegistryError):
        old_store.recipient_key("hospital-B", "B-x25519-2026a")    # cannot keep serving v1
    with pytest.raises(RegistryError):
        new_store.recipient_key("hospital-B", "B-x25519-2026a")    # revoked in v2
    new_store.signing_key("hospital-A", "A-ed25519-2026a")         # latest valid state still works


def test_af052_unloaded_store_and_registry_object_have_no_lookup_path(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(RegistryError):
        store.recipient_key("hospital-B", "B-x25519-2026a")
    reg = store.load(_snap(1), now=NOW)
    assert not hasattr(reg, "recipient_key") and not hasattr(reg, "signing_key")
