"""Signed key registry with pinned root and persisted anti-rollback (§8.6, opinion F1).

Snapshot bytes::

    payload  = LP("soteria-share/registry/v1") || u64 version || i64 issued_at || i64 expires_at
               || u32 n || record_1 .. record_n           (records strictly sorted by (inst_id, key_id))
    record   = LP(inst_id) || LP(key_id) || LP(purpose) || LP(status) || LP(public_key[32])
    snapshot = LP(payload) || sig(64)                      sig = Ed25519_root(LP(SIG_DOMAIN) || payload)

The root public key is pinned by configuration and never read from a snapshot.
A snapshot is accepted only if its signature and every validity rule hold; only
then is the highest-seen (version, payload digest) persisted atomically, so a
failed or unverified snapshot never changes trusted state. Lookups are served
only by :class:`TrustStore`, which re-checks the clock and the persisted
high-water mark on every call, so a long-lived service cannot keep using an
expired or superseded snapshot (AF-052). This is a bounded
anti-rollback mechanism for the prototype, not a full federation PKI.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey

from .encoding import Reader, decode_id, lp, require_id
from .errors import RegistryError, ShareFormatError


PAYLOAD_DOMAIN = b"soteria-share/registry/v1"
SIG_DOMAIN = b"soteria-share/registry-sig/v1"
PURPOSES = ("x25519-recv", "ed25519-sign")
STATUSES = ("active", "revoked")
MAX_SNAPSHOT_BYTES = 1 << 20
MAX_RECORDS = 10_000


@dataclass(frozen=True)
class KeyRecord:
    inst_id: str
    key_id: str
    purpose: str
    status: str
    public_key: bytes

    def encode(self) -> bytes:
        require_id(self.inst_id, "inst_id")
        require_id(self.key_id, "key_id")
        if self.purpose not in PURPOSES or self.status not in STATUSES or len(self.public_key) != 32:
            raise ShareFormatError("invalid registry record")
        return b"".join(lp(x) for x in (self.inst_id.encode(), self.key_id.encode(), self.purpose.encode(),
                                         self.status.encode(), self.public_key))


@dataclass(frozen=True)
class Registry:
    """Verified snapshot data. Key lookups go only through :class:`TrustStore` (AF-052)."""

    version: int
    issued_at: int
    expires_at: int
    records: tuple[KeyRecord, ...]
    digest: str

    def _find(self, inst_id: str, key_id: str, purpose: str) -> KeyRecord:
        for record in self.records:
            if record.inst_id == inst_id and record.key_id == key_id:
                if record.purpose != purpose:
                    raise RegistryError("registry key has a different purpose")
                if record.status != "active":
                    raise RegistryError("registry key is not active")
                return record
        raise RegistryError("unknown registry key")



def build_payload(version: int, issued_at: int, expires_at: int, records: list[KeyRecord]) -> bytes:
    ordered = sorted(records, key=lambda r: (r.inst_id, r.key_id))
    return (lp(PAYLOAD_DOMAIN) + struct.pack(">QqqI", version, issued_at, expires_at, len(ordered))
            + b"".join(r.encode() for r in ordered))


def build_snapshot(root: Ed25519PrivateKey, version: int, issued_at: int, expires_at: int,
                   records: list[KeyRecord]) -> bytes:
    """Authority-side helper (used by tests and the CPU demo)."""
    payload = build_payload(version, issued_at, expires_at, records)
    return lp(payload) + root.sign(lp(SIG_DOMAIN) + payload)


def _parse_payload(payload: bytes) -> tuple[int, int, int, tuple[KeyRecord, ...]]:
    reader = Reader(payload)
    if reader.lp(64) != PAYLOAD_DOMAIN:
        raise ShareFormatError("not a registry payload")
    version, issued_at, expires_at, count = struct.unpack(">QqqI", reader.fixed(28))
    if count > MAX_RECORDS:
        raise ShareFormatError("too many registry records")
    records = []
    for _ in range(count):
        inst_id = decode_id(reader.lp(64), "inst_id")
        key_id = decode_id(reader.lp(64), "key_id")
        purpose = reader.lp(32).decode("ascii", "replace")
        status = reader.lp(32).decode("ascii", "replace")
        public_key = reader.lp(32, exact=32)
        if purpose not in PURPOSES or status not in STATUSES:
            raise ShareFormatError("invalid registry record purpose/status")
        records.append(KeyRecord(inst_id, key_id, purpose, status, public_key))
    reader.done()
    keys = [(r.inst_id, r.key_id) for r in records]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ShareFormatError("registry records must be strictly sorted and unique")
    return version, issued_at, expires_at, tuple(records)


class TrustStore:
    """Pinned root + persisted highest-seen (version, digest); shared by sender and receiver."""

    def __init__(self, root: Ed25519PublicKey, state_path: Path, clock: Callable[[], float] | None = None):
        if not isinstance(root, Ed25519PublicKey):
            raise ShareFormatError("pinned root must be an Ed25519 public key")
        self._root = root
        self._state_path = Path(state_path)
        self._clock = clock if clock is not None else (lambda: time.time())
        self._current: Registry | None = None

    # ---- use-time resolution (AF-052): every lookup re-checks clock and persisted high-water
    def _checked_current(self) -> Registry:
        current = self._current
        if current is None:
            raise RegistryError("no verified registry snapshot is loaded")
        now = int(self._clock())
        if not current.issued_at <= now < current.expires_at:
            raise RegistryError("loaded registry snapshot is expired or not yet valid")
        state = self._read_state()
        if state is None or state != (current.version, current.digest):
            raise RegistryError("loaded registry snapshot is stale relative to the persisted high-water mark")
        return current

    def recipient_key(self, inst_id: str, key_id: str) -> X25519PublicKey:
        record = self._checked_current()._find(inst_id, key_id, "x25519-recv")
        return X25519PublicKey.from_public_bytes(record.public_key)

    def signing_key(self, inst_id: str, key_id: str) -> Ed25519PublicKey:
        record = self._checked_current()._find(inst_id, key_id, "ed25519-sign")
        return Ed25519PublicKey.from_public_bytes(record.public_key)

    @property
    def current(self) -> Registry | None:
        """Read-only view of the last verified snapshot (inspection only; never used for lookups)."""
        return self._current

    @property
    def records(self) -> tuple[KeyRecord, ...]:
        return self._current.records if self._current is not None else ()

    def _read_state(self) -> tuple[int, str] | None:
        if not self._state_path.exists():
            return None
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            return int(state["highest_version"]), str(state["digest"])
        except (ValueError, KeyError, TypeError) as exc:
            raise RegistryError("trusted registry state is unreadable; failing closed") from exc

    def _write_state(self, version: int, digest: str) -> None:
        tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump({"highest_version": version, "digest": digest}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self._state_path)

    def load(self, snapshot: bytes, *, now: int | None = None) -> Registry:
        """Verify fully, then (and only then) advance the persisted anti-rollback state and adopt it."""
        now = int(self._clock()) if now is None else int(now)
        try:
            if not isinstance(snapshot, (bytes, bytearray)) or len(snapshot) > MAX_SNAPSHOT_BYTES:
                raise ShareFormatError("registry snapshot missing or too large")
            reader = Reader(snapshot)
            payload = reader.lp(MAX_SNAPSHOT_BYTES)
            sig = reader.fixed(64)
            reader.done()
        except ShareFormatError as exc:
            raise RegistryError("malformed registry snapshot") from exc
        try:
            self._root.verify(sig, lp(SIG_DOMAIN) + payload)
        except InvalidSignature:
            raise RegistryError("registry signature does not verify under the pinned root") from None
        try:
            version, issued_at, expires_at, records = _parse_payload(payload)
        except ShareFormatError as exc:
            raise RegistryError("registry payload is malformed") from exc
        if not issued_at <= now < expires_at:
            raise RegistryError("registry snapshot is expired or not yet valid")
        digest = hashlib.sha256(payload).hexdigest()
        state = self._read_state()
        if state is not None:
            highest, seen_digest = state
            if version < highest:
                raise RegistryError("registry rollback: older than the highest version seen")
            if version == highest and digest != seen_digest:
                raise RegistryError("registry equivocation: same version, different content")
        if state is None or version > state[0]:
            self._write_state(version, digest)
        registry = Registry(version, issued_at, expires_at, records, digest)
        self._current = registry
        return registry
