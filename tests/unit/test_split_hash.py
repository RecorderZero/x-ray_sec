from scripts.create_dev_split import stable_split_hash


def test_split_hash_ignores_local_mount_path() -> None:
    record = {
        "sample_id": "dev_v1.1_000",
        "patient_id": "patient00001",
        "label": 0,
        "source_path": "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg",
        "local_path": "/mnt/a/view1_frontal.jpg",
        "file_sha256": "a" * 64,
    }
    moved = dict(record, local_path="/another/machine/view1_frontal.jpg")
    assert stable_split_hash([record]) == stable_split_hash([moved])
