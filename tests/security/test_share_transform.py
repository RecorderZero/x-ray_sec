"""S3 / P1: pure transform round trip (no PNG), pre-registered max-abs < 1e-5.

Fixtures (§8.8 P1): d=65536, float32; the first 16 cached real latents in
artifacts/t2bb_latent_cache_dpm (if present on this machine) plus 16 seed-1911
N(0, I) latents. The bound is claimed only for these fixtures.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from src.crypto.kdf import derive_subkey, round_label
from src.crypto.s2_fixed_norm import FIXED_NORM
from src.crypto.sot import transform as v1_transform
from src.share.encoding import PROFILE_M1_PRIVATE_RANGE_V2 as PROFILE
from src.share.errors import ShareFormatError
from src.share.keys import SecretKey, TestFixtureStream, derive_subkeys, wipe_all
from src.share.transform import LATENT_SHAPE, decrypt_latent, dphd, dphd_inverse, encrypt_latent


TOL = 1e-5
CACHE = Path(__file__).resolve().parents[2] / "artifacts/t2bb_latent_cache_dpm"


def _gaussian_fixtures() -> list[torch.Tensor]:
    rng = np.random.default_rng(1911)
    return [torch.from_numpy(rng.standard_normal(65_536).astype(np.float32)).reshape(LATENT_SHAPE) for _ in range(16)]


def _real_fixtures() -> list[torch.Tensor]:
    index = CACHE / "index.csv"
    if not index.exists():
        return []
    with index.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:16]
    return [torch.from_numpy(np.load(CACHE / "latents" / f"{r['file_sha256']}.npy").astype(np.float32)).reshape(LATENT_SHAPE)
            for r in rows]


def _subkeys(label: str):
    stream = TestFixtureStream("transform/" + label)
    key, nonce = stream.image_key(), stream.nonce()
    subkeys = derive_subkeys(key, nonce, PROFILE)
    key.wipe()
    return subkeys


@pytest.mark.parametrize("source", ["gaussian", "real"])
def test_p1_pure_round_trip_within_preregistered_tolerance(source: str) -> None:
    fixtures = _gaussian_fixtures() if source == "gaussian" else _real_fixtures()
    if not fixtures:
        pytest.skip("real latent cache not present on this machine")
    worst = 0.0
    for i, z in enumerate(fixtures):
        subkeys = _subkeys(f"{source}/{i}")
        c, norm = encrypt_latent(z, subkeys)
        assert c.dtype == torch.float32 and tuple(c.shape) == LATENT_SHAPE
        assert abs(float(torch.linalg.vector_norm(c.double())) - float(FIXED_NORM)) < 1e-3
        back = decrypt_latent(c, subkeys, norm)
        worst = max(worst, float((back - z).abs().max()))
        wipe_all(subkeys)
    assert worst < TOL, f"P1 worst max-abs {worst:.3e} >= {TOL}"


def test_dphd_is_orthogonal_on_basis_probes() -> None:
    subkeys = _subkeys("orth")
    rng = np.random.default_rng(7)
    a = torch.from_numpy(rng.standard_normal(65_536).astype(np.float32)).reshape(LATENT_SHAPE)
    b = torch.from_numpy(rng.standard_normal(65_536).astype(np.float32)).reshape(LATENT_SHAPE)
    qa, qb = dphd(a, subkeys), dphd(b, subkeys)
    assert abs(float((qa.double() * qb.double()).sum()) - float((a.double() * b.double()).sum())) < 1e-2
    assert float((dphd_inverse(qa, subkeys) - a).abs().max()) < TOL
    wipe_all(subkeys)


def test_v2_math_equals_v1_when_fed_v1_subkeys() -> None:
    """Same D/P/D streams => bit-identical output to the frozen v1 transform."""
    master, nonce = bytes(range(32)), bytes(range(16))
    subkeys = {
        "round/0/sign-pre": SecretKey(derive_subkey(master, nonce, round_label(0, "sign-pre"))),
        "round/0/perm": SecretKey(derive_subkey(master, nonce, round_label(0, "perm"))),
        "round/0/sign-post": SecretKey(derive_subkey(master, nonce, round_label(0, "sign-post"))),
    }
    x = _gaussian_fixtures()[0]
    assert torch.equal(dphd(x, subkeys), v1_transform(x, master, nonce, rounds=1, use_permutation=True))


def test_different_recipients_give_different_ciphertexts() -> None:
    z = _gaussian_fixtures()[1]
    cb, _ = encrypt_latent(z, _subkeys("B"))
    cc, _ = encrypt_latent(z, _subkeys("C"))
    cos = float((cb.double() * cc.double()).sum() / (cb.double().norm() * cc.double().norm()))
    assert abs(cos) < 0.05  # smoke check on a non-degenerate fixture only (N8)


@pytest.mark.parametrize("bad", [
    torch.zeros(1, 1, 128, 128), torch.zeros(1, 1, 256, 256, dtype=torch.float64),
    torch.full((1, 1, 256, 256), float("nan")),
])
def test_rejects_bad_latents(bad: torch.Tensor) -> None:
    with pytest.raises(ShareFormatError):
        encrypt_latent(bad, _subkeys("bad"))


def test_rejects_bad_norm() -> None:
    subkeys = _subkeys("norm")
    c, _ = encrypt_latent(_gaussian_fixtures()[2], subkeys)
    for norm in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ShareFormatError):
            decrypt_latent(c, subkeys, norm)
