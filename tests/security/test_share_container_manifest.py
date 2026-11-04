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


# ------------------------------------------------------------------ AF-057 residual: backfill binding + counts
def _backfill_world(tmp_path, split_rows, ref_split_sha: str, n_ref: int | None = None):
    import json

    from scripts.soteria_share_experiments import _aggregate
    world = World(tmp_path)
    art = tmp_path / "art"
    blobs = {}
    for role_dir, recipient, tagc in (("e1_smoke_n4/containers_B", world.b, "B"), ("e2_smoke_n4/containers_C", world.c, "C")):
        d = art / role_dir
        d.mkdir(parents=True)
        bl = []
        for i, r in enumerate(split_rows):
            blob = share_image(x0_fixture(i), sender=world.sender, recipient_inst_id=recipient.inst_id,
                               recipient_key_id=recipient.key_id, registry=world.registry, manifest_digest=MANIFEST.digest(),
                               invert=identity, generate=identity, fixture=TestFixtureStream(f"bf/{tagc}/{r['sample_id']}"))
            (d / f"{r['sample_id']}.soteria").write_bytes(blob)
            bl.append(blob)
        blobs[tagc] = bl
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"experiment": "E2", "evidence_role": "smoke", "split_sha256": ref_split_sha,
                               "n_patients": n_ref if n_ref is not None else len(split_rows),
                               "b_containers_sha256": _aggregate(blobs["B"]), "c_containers_sha256": _aggregate(blobs["C"])}))
    return art, ref


def _write_split(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    return path


def test_backfill_refuses_split_whose_patient_ids_differ_from_reference(tmp_path) -> None:
    import types

    from scripts.rq4_formal_redo import sha256_file
    from scripts.soteria_share_experiments import run_backfill
    rows = list(csv.DictReader(SPLIT.open(encoding="utf-8")))[:4]
    good_split = _write_split(tmp_path / "good.csv", [{**r} for r in rows] + list(csv.DictReader(SPLIT.open(encoding="utf-8")))[4:])
    art, ref = _backfill_world(tmp_path, rows, sha256_file(good_split))
    bad_rows = [{**rows[0], "patient_id": "AUDITOR_WRONG_PATIENT"}] + list(csv.DictReader(SPLIT.open(encoding="utf-8")))[1:]
    bad_split = _write_split(tmp_path / "bad.csv", bad_rows)
    args = types.SimpleNamespace(split=bad_split, evidence_role="smoke", limit=4, artifact_dir=art, reference=ref)
    with pytest.raises(SystemExit):
        run_backfill(args)
    assert not (art / "e1_smoke_n4/containers_B/container_manifest.json").exists()
    assert not (art / "e2_smoke_n4/containers_C/container_manifest.json").exists()
    # the genuine split still backfills (labelled retroactive)
    run_backfill(types.SimpleNamespace(split=good_split, evidence_role="smoke", limit=4, artifact_dir=art, reference=ref))
    import json
    man = json.loads((art / "e1_smoke_n4/containers_B/container_manifest.json").read_text())
    assert man["created_by"].startswith("retroactive")


@pytest.mark.parametrize("problem", ["role", "count"])
def test_backfill_refuses_reference_role_or_count_mismatch(tmp_path, problem) -> None:
    import json
    import types

    from scripts.rq4_formal_redo import sha256_file
    from scripts.soteria_share_experiments import run_backfill
    rows = list(csv.DictReader(SPLIT.open(encoding="utf-8")))[:4]
    art, ref = _backfill_world(tmp_path, rows, sha256_file(SPLIT), n_ref=5 if problem == "count" else None)
    if problem == "role":
        data = json.loads(ref.read_text()); data["evidence_role"] = "formal"; ref.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        run_backfill(types.SimpleNamespace(split=SPLIT, evidence_role="smoke", limit=4, artifact_dir=art, reference=ref))
    assert not (art / "e1_smoke_n4/containers_B/container_manifest.json").exists()


def test_loader_rejects_short_entry_list_and_writer_rejects_length_mismatch(tmp_path) -> None:
    import json
    rows, d = _setup(tmp_path)
    man_path = d / "container_manifest.json"
    man = json.loads(man_path.read_text())
    man["entries"] = man["entries"][:3]            # n still says 4
    man_path.write_text(json.dumps(man))
    with pytest.raises(SystemExit):
        load_verified_containers(d, "E1-B", rows, SPLIT, "hospital-B")
    with pytest.raises(SystemExit):
        write_container_manifest(d, "E1-B", rows, [b"x"] * 3, SPLIT, created_by="test")
