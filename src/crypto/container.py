"""D5.5 strict canonical authenticated-container schema.

The MAC input is canonical UTF-8 JSON with sorted keys and compact separators.
Floating-point values are forbidden: M1 lo/hi are authenticated as their exact
float32 IEEE-754 hex bit patterns. The ciphertext bytes themselves are covered,
not merely a reference or digest.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from PIL import Image

from scripts.m1_storage import HI_KEY, LO_KEY, RANGE_PRESERVING_PNG

from .errors import CryptoConfigurationError
from .kdf import NONCE_BYTES, require_nonce


VERSION = "cxr-auth-container/v1"
TAG_BYTES = 32
TOP_LEVEL_FIELDS = {"version", "nonce", "parameters", "aad", "storage", "ciphertext", "tag"}
AAD_FIELDS = {
    "study_instance_uid", "sop_instance_uid", "shape", "dtype",
    "model_sha256", "sampler_version",
}
STORAGE_FIELDS = {
    "mode", "storage_protocol", "quantization", "lo_float32_bits",
    "hi_float32_bits", "payload_sha256",
}
HEX8 = re.compile(r"^[0-9a-f]{8}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _no_duplicate_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise CryptoConfigurationError("duplicate JSON field")
        output[key] = value
    return output


def _validate_json_value(value: Any, path: str = "parameters") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _validate_json_value(item, f"{path}.{key}")
        return
    raise CryptoConfigurationError(f"{path} contains a non-canonical type")


def _strict_b64decode(text: str) -> bytes:
    if not isinstance(text, str):
        raise CryptoConfigurationError("base64 field must be text")
    try:
        output = base64.b64decode(text, validate=True)
    except Exception as error:
        raise CryptoConfigurationError("invalid base64") from error
    if base64.b64encode(output).decode("ascii") != text:
        raise CryptoConfigurationError("non-canonical base64")
    return output


@dataclass(frozen=True)
class AuthenticatedContainer:
    nonce: bytes
    parameters: Mapping[str, Any]
    aad: Mapping[str, Any]
    storage: Mapping[str, Any]
    ciphertext: bytes
    tag: bytes = b""
    version: str = VERSION

    def validate(self, *, require_tag: bool) -> None:
        if self.version != VERSION:
            raise CryptoConfigurationError("unknown container version")
        require_nonce(self.nonce)
        if not isinstance(self.ciphertext, bytes) or not self.ciphertext:
            raise CryptoConfigurationError("ciphertext must be non-empty bytes")
        if not isinstance(self.parameters, Mapping):
            raise CryptoConfigurationError("parameters must be an object")
        _validate_json_value(dict(self.parameters))
        if set(self.aad) != AAD_FIELDS:
            raise CryptoConfigurationError("AAD schema mismatch")
        _validate_json_value(dict(self.aad), "aad")
        shape = self.aad["shape"]
        if not isinstance(shape, list) or not shape or not all(isinstance(v, int) and v > 0 for v in shape):
            raise CryptoConfigurationError("AAD shape must be positive integers")
        if not HEX64.fullmatch(str(self.aad["model_sha256"])):
            raise CryptoConfigurationError("invalid model hash")
        if set(self.storage) != STORAGE_FIELDS:
            raise CryptoConfigurationError("storage schema mismatch")
        _validate_json_value(dict(self.storage), "storage")
        if self.storage["mode"] not in {"M1", "M2"}:
            raise CryptoConfigurationError("unknown storage mode")
        if not HEX64.fullmatch(str(self.storage["payload_sha256"])):
            raise CryptoConfigurationError("invalid payload hash")
        if self.storage["payload_sha256"] != hashlib.sha256(self.ciphertext).hexdigest():
            raise CryptoConfigurationError("payload hash mismatch")
        if self.storage["mode"] == "M1":
            if self.storage["storage_protocol"] != RANGE_PRESERVING_PNG:
                raise CryptoConfigurationError("unknown M1 storage protocol")
            if not HEX8.fullmatch(str(self.storage["lo_float32_bits"])) or not HEX8.fullmatch(
                str(self.storage["hi_float32_bits"])
            ):
                raise CryptoConfigurationError("M1 lo/hi must be exact float32 bit hex")
        elif self.storage["lo_float32_bits"] or self.storage["hi_float32_bits"]:
            raise CryptoConfigurationError("M2 must not carry M1 lo/hi")
        if require_tag and (not isinstance(self.tag, bytes) or len(self.tag) != TAG_BYTES):
            raise CryptoConfigurationError("tag must be exactly 32 bytes")
        if not require_tag and self.tag:
            raise CryptoConfigurationError("unsigned container must have an empty tag")

    def _document(self, *, include_tag: bool) -> dict[str, Any]:
        document = {
            "version": self.version,
            "nonce": self.nonce.hex(),
            "parameters": dict(self.parameters),
            "aad": dict(self.aad),
            "storage": dict(self.storage),
            "ciphertext": base64.b64encode(self.ciphertext).decode("ascii"),
        }
        if include_tag:
            document["tag"] = self.tag.hex()
        return document

    def mac_bytes(self) -> bytes:
        self.validate(require_tag=bool(self.tag))
        return json.dumps(
            self._document(include_tag=False), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def serialize(self) -> bytes:
        self.validate(require_tag=True)
        return (json.dumps(
            self._document(include_tag=True), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        ) + "\n").encode("utf-8")

    def with_tag(self, tag: bytes) -> "AuthenticatedContainer":
        return replace(self, tag=tag)


def deserialize_container(payload: bytes) -> AuthenticatedContainer:
    if not isinstance(payload, bytes):
        raise CryptoConfigurationError("serialized container must be bytes")
    try:
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=_no_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CryptoConfigurationError("invalid container encoding") from error
    if not isinstance(document, dict) or set(document) != TOP_LEVEL_FIELDS:
        raise CryptoConfigurationError("container top-level schema mismatch")
    try:
        nonce = bytes.fromhex(document["nonce"])
        tag = bytes.fromhex(document["tag"])
    except (TypeError, ValueError) as error:
        raise CryptoConfigurationError("invalid nonce or tag encoding") from error
    container = AuthenticatedContainer(
        version=document["version"], nonce=nonce, parameters=document["parameters"],
        aad=document["aad"], storage=document["storage"],
        ciphertext=_strict_b64decode(document["ciphertext"]), tag=tag,
    )
    container.validate(require_tag=True)
    return container


def validate_m1_png(container: AuthenticatedContainer) -> bytes:
    """Post-authentication M1 consistency check; never call before MAC verify."""
    if container.storage["mode"] != "M1":
        raise CryptoConfigurationError("container is not M1")
    with Image.open(io.BytesIO(container.ciphertext)) as image:
        if image.mode != "L":
            raise CryptoConfigurationError("M1 payload is not 8-bit grayscale PNG")
        text = dict(image.text)
    if text.get(LO_KEY) != container.storage["lo_float32_bits"] or text.get(HI_KEY) != container.storage[
        "hi_float32_bits"
    ]:
        raise CryptoConfigurationError("PNG and authenticated lo/hi differ")
    return container.ciphertext


def make_storage_record(
    *, mode: str, storage_protocol: str, quantization: str,
    lo_float32_bits: str, hi_float32_bits: str, payload: bytes,
) -> dict[str, str]:
    return {
        "mode": mode,
        "storage_protocol": storage_protocol,
        "quantization": quantization,
        "lo_float32_bits": lo_float32_bits,
        "hi_float32_bits": hi_float32_bits,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
