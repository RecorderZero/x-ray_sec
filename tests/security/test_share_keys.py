"""S2: v2 encoding + per-recipient image key (V3 library level, V6 freshness, §8.4.1)."""

import pickle

import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from src.crypto.kdf import derive_subkey, round_label
from src.share.encoding import PROFILE_M1_PRIVATE_RANGE_V2 as PROFILE
from src.share.encoding import Reader, lp, lp_join, require_id, require_profile
from src.share.errors import ShareFormatError, UnwrapError
from src.share.keys import (
    SUBKEY_LABELS, WRAP_BYTES, SecretKey, TestFixtureStream, build_info, derive_subkeys,
    fixture_x25519, new_image_key, new_nonce, unwrap_image_key, wipe_all, wrap_image_key,
)


IDS = dict(inst_id_b="hospital-B", key_id_b="B-x25519-2026a", inst_id_a="hospital-A", signing_key_id_a="A-ed25519-2026a")


def _info(nonce: bytes, **override) -> bytes:
    fields = {**IDS, **override}
    return build_info(PROFILE, nonce, fields["inst_id_b"], fields["key_id_b"], fields["inst_id_a"], fields["signing_key_id_a"])


# ---------------------------------------------------------------- encoding (§8.4.1 (1))
def test_lp_round_trip_and_strict_reader() -> None:
    blob = lp_join(b"abc", b"", b"\x00" * 5)
    reader = Reader(blob)
    assert reader.lp(16) == b"abc" and reader.lp(16) == b"" and reader.lp(16) == b"\x00" * 5
    reader.done()


def test_lp_is_unambiguous() -> None:
    assert lp_join(b"ab", b"c") != lp_join(b"a", b"bc")


@pytest.mark.parametrize("blob", [
    b"\x00\x00\x00",                       # truncated length prefix
    b"\x00\x00\x00\x05abc",                # length runs past buffer
    b"\xff\xff\xff\xffabc",                # overflow-sized length
])
def test_reader_rejects_truncation_and_overflow(blob: bytes) -> None:
    with pytest.raises(ShareFormatError):
        Reader(blob).lp(1 << 20)


def test_reader_rejects_trailing_bytes_and_over_limit() -> None:
    reader = Reader(lp(b"abc") + b"x")
    reader.lp(8)
    with pytest.raises(ShareFormatError):
        reader.done()
    with pytest.raises(ShareFormatError):
        Reader(lp(b"abcdef")).lp(3)
    with pytest.raises(ShareFormatError):
        Reader(lp(b"abc")).lp(8, exact=4)


@pytest.mark.parametrize("bad", ["", "soteria-share/m1-private-range", "SOTERIA-share/x/v2", "soteria-share/other/v2", 7])
def test_profile_allowlist(bad) -> None:
    with pytest.raises(ShareFormatError):
        require_profile(bad)


@pytest.mark.parametrize("bad", ["", "a b", "x" * 65, "病院", "id/with/slash"])
def test_id_validation(bad: str) -> None:
    with pytest.raises(ShareFormatError):
        require_id(bad, "id")


# ---------------------------------------------------------------- info (§8.4.1 (4))
def test_info_binds_every_field() -> None:
    nonce = bytes(range(16))
    base = _info(nonce)
    assert base == _info(nonce)
    assert base.startswith(lp(PROFILE.encode()))
    variants = [_info(bytes(16))] + [_info(nonce, **{k: v + "x"}) for k, v in IDS.items()]
    assert len({base, *variants}) == len(variants) + 1


# ---------------------------------------------------------------- HPKE wrap (V3 library level)
def test_wrap_round_trip() -> None:
    sk = fixture_x25519("B")
    nonce = new_nonce()
    with new_image_key() as key:
        wrap = wrap_image_key(key, sk.public_key(), _info(nonce))
        assert len(wrap) == WRAP_BYTES
        with unwrap_image_key(wrap, sk, _info(nonce)) as back:
            assert back == key


def test_unwrap_fails_closed_on_wrong_key_info_or_malformed_wrap() -> None:
    sk = fixture_x25519("B")
    nonce = bytes(range(16))
    with new_image_key() as key:
        wrap = wrap_image_key(key, sk.public_key(), _info(nonce))
    cases = [
        (wrap, X25519PrivateKey.generate(), _info(nonce)),                       # wrong private key
        (wrap, fixture_x25519("C"), _info(nonce)),                                # another recipient
        (wrap, sk, _info(bytes(16))),                                             # wrong nonce in info
        (wrap, sk, _info(nonce, key_id_b="B-x25519-old")),                        # wrong recipient key id
        (wrap, sk, _info(nonce, signing_key_id_a="A-ed25519-other")),             # wrong sender binding
        (wrap[:-1], sk, _info(nonce)),                                            # truncated
        (wrap + b"\x00", sk, _info(nonce)),                                       # extended
        (bytes([wrap[0] ^ 1]) + wrap[1:], sk, _info(nonce)),                      # enc bit flip
        (wrap[:-1] + bytes([wrap[-1] ^ 1]), sk, _info(nonce)),                    # tag bit flip
        (bytes(WRAP_BYTES), sk, _info(nonce)),                                    # all-zero
    ]
    for wrap_i, key_i, info_i in cases:
        with pytest.raises(UnwrapError):
            unwrap_image_key(wrap_i, key_i, info_i)


# ---------------------------------------------------------------- freshness (V6)
def test_fresh_keys_nonces_and_wraps() -> None:
    sk = fixture_x25519("B")
    keys = [new_image_key() for _ in range(8)]
    assert len({k.raw() for k in keys}) == 8
    assert len({new_nonce() for _ in range(8)}) == 8
    nonce = bytes(16)
    wraps = {wrap_image_key(keys[0], sk.public_key(), _info(nonce)) for _ in range(4)}
    assert len(wraps) == 4  # HPKE ephemeral key is fresh per call
    wipe_all(*keys)


# ---------------------------------------------------------------- subkeys (§8.4.1 (5))
def test_subkeys_are_distinct_deterministic_and_domain_separated() -> None:
    stream = TestFixtureStream("subkeys")
    key, nonce = stream.image_key(), stream.nonce()
    first = derive_subkeys(key, nonce, PROFILE)
    again = derive_subkeys(key, nonce, PROFILE)
    assert set(first) == set(SUBKEY_LABELS)
    assert len({k.raw() for k in first.values()}) == len(SUBKEY_LABELS)
    assert all(first[label] == again[label] for label in SUBKEY_LABELS)
    other_nonce = derive_subkeys(key, bytes(16), PROFILE)
    assert all(first[label] != other_nonce[label] for label in SUBKEY_LABELS)
    # v1 derivation of the same raw bytes is a different domain
    v1 = derive_subkey(key.raw(), nonce, round_label(0, "sign-pre"))
    assert v1 != first["round/0/sign-pre"].raw()
    wipe_all(key, first, again, other_nonce)


# ---------------------------------------------------------------- buffer hygiene (§8.5)
def test_secret_key_wipes_redacts_and_refuses_pickle() -> None:
    key = new_image_key()
    assert "redacted" in repr(key) and key.raw().hex() not in repr(key) and key.raw().hex() not in str(key)
    with pytest.raises(TypeError):
        pickle.dumps(key)
    key.wipe()
    assert key.wiped and key.buffer_is_zero()
    with pytest.raises(ShareFormatError):
        key.raw()


def test_context_manager_wipes_on_exception() -> None:
    key = new_image_key()
    with pytest.raises(RuntimeError):
        with key:
            raise RuntimeError("boom")
    assert key.wiped and key.buffer_is_zero()


def test_fixture_stream_is_deterministic_and_label_separated() -> None:
    assert TestFixtureStream("a").read(100) == TestFixtureStream("a").read(100)
    assert TestFixtureStream("a").read(32) != TestFixtureStream("b").read(32)
    assert fixture_x25519("B").public_key().public_bytes_raw() == fixture_x25519("B").public_key().public_bytes_raw()
