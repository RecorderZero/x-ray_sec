"""AF-057: E2 reuse of E1 containers is verified per sample against a private manifest."""

import csv
from pathlib import Path

import pytest

from scripts.soteria_share_experiments import load_verified_containers, write_container_manifest
from src.share.keys import TestFixtureStream
from src.share.sender import share_image
from tests.security._share_world import MANIFEST, World, identity, x0_fixture

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "splits/security_v1.csv"
OTHER_SPLIT = ROOT / "splits/t2bb_inner_n200_v1.csv"


def _setup(tmp_path):
    world = World(tmp_path)
    rows = list(csv.DictReader(SPLIT.open(encoding="utf-8")))[:4]
    d = tmp_path / "containers_B"
    d.mkdir()
    blobs = []
    for i, r in enumerate(rows):
        blob = share_image(x0_fixture(i), sender=world.sender, recipient_inst_id=world.b.inst_id,
                           recipient_key_id=world.b.key_id, registry=world.registry, manifest_digest=MANIFEST.digest(),
                           invert=identity, generate=identity, fixture=TestFixtureStream(f"manifest/{r['sample_id']}"))
        (d / f"{r['sample_id']}.soteria").write_bytes(blob)
        blobs.append(blob)
    write_container_manifest(d, "E1-B", rows, blobs, SPLIT, created_by="test")
    return rows, d


def test_unmodified_containers_are_accepted(tmp_path) -> None:
    rows, d = _setup(tmp_path)
    blobs, source = load_verified_containers(d, "E1-B", rows, SPLIT, "hospital-B")
    assert len(blobs) == 4 and source["created_by"] == "test"


def test_swapped_containers_are_refused(tmp_path) -> None:
    rows, d = _setup(tmp_path)
    a, b = d / f"{rows[0]['sample_id']}.soteria", d / f"{rows[1]['sample_id']}.soteria"
    x, y = a.read_bytes(), b.read_bytes()
    a.write_bytes(y)
    b.write_bytes(x)
    with pytest.raises(SystemExit):
        load_verified_containers(d, "E1-B", rows, SPLIT, "hospital-B")


@pytest.mark.parametrize("case", ["other_split", "edited_blob", "wrong_role", "wrong_recipient", "reordered_rows", "missing_manifest"])
def test_other_mismatches_are_refused(tmp_path, case: str) -> None:
    rows, d = _setup(tmp_path)
    split, role, recipient = SPLIT, "E1-B", "hospital-B"
    if case == "other_split":
        split = OTHER_SPLIT
    elif case == "edited_blob":
        f = d / f"{rows[2]['sample_id']}.soteria"
        data = bytearray(f.read_bytes())
        data[200] ^= 1
        f.write_bytes(bytes(data))
    elif case == "wrong_role":
        role = "E2-C"
    elif case == "wrong_recipient":
        recipient = "hospital-C"
    elif case == "reordered_rows":
        rows = [rows[1], rows[0], *rows[2:]]
    else:
        (d / "container_manifest.json").unlink()
    with pytest.raises(SystemExit):
        load_verified_containers(d, role, rows, split, recipient)
