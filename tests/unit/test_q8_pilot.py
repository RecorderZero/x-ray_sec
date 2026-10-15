"""CPU-only tests for the Q8/P8 pilot runner, analyzer and metric helpers.

The runner is exercised with ``SyntheticBackend`` (a fake pointwise generator), so
these tests prove plumbing, schema, controls, hashing, determinism and fail-closed
behaviour -- never any scientific value of the diffusion pipeline.
"""

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import e2_legacy_wrapper as legacy
from scripts import q8_analyze, q8_metrics, q8_pilot_runner as runner
from scripts.e2_anonymization_runner import legacy_png_handoff
from scripts.q8_metrics import (
    LpipsUnavailable, container_storage_breakdown, lpips_status, nn_ranks, parse_col, select_visual_cases,
    shuffled_truth, summarize, timing_summary,
)
from src.crypto.nonce import NonceRegistry
from src.crypto.s2_fixed_norm import seal_anonymous_image, seal_latent

KEY = bytes(range(32))


# ----------------------------------------------------------------- metric helpers
def test_bootstrap_is_deterministic_and_clusters_by_patient() -> None:
    values = [1.0, 2.0, 3.0, 10.0]
    assert summarize(values) == summarize(values)
    clustered = summarize(values, ["a", "a", "b", "b"])
    assert clustered["n"] == 4 and clustered["mean"] == 4.0
    assert summarize(values, ["a", "a", "b", "b"])["ci95_high"] != summarize(values)["ci95_high"]
    with pytest.raises(ValueError):
        summarize([1.0, float("nan")])


def test_timing_summary_median_p95_linear() -> None:
    stats = timing_summary(np.arange(1, 101, dtype=float))
    assert stats["median"] == 50.5 and stats["p95"] == pytest.approx(95.05)
    with pytest.raises(ValueError):
        timing_summary([-1.0])


def test_nn_rank_pessimistic_ties_and_shuffled_derangement() -> None:
    same = np.ones((3, 4))
    assert nn_ranks(same, same, "cosine_raw").tolist() == [3, 3, 3]  # truth last among exact ties
    gallery = np.eye(4)
    assert nn_ranks(gallery, gallery, "pearson").tolist() == [1, 1, 1, 1]
    for size in (2, 5, 50):
        truth = shuffled_truth(size)
        assert sorted(truth) == list(range(size)) and all(truth[i] != i for i in range(size))
    assert shuffled_truth(7).tolist() == shuffled_truth(7).tolist()


def test_visual_selection_is_rule_based_without_substitution() -> None:
    ids = [f"s{i}" for i in range(5)]
    labels = ["0", "1", "0", "1", "0"]
    scores = [30.0, 29.0, 25.0, 31.0, 25.0]
    cases = {case["role"]: case for case in select_visual_cases(ids, labels, scores)}
    assert cases["fixed_label0"]["sample_id"] == "s0" and cases["fixed_label1"]["sample_id"] == "s1"
    assert cases["worst"]["sample_id"] == "s2"  # tie with s4 broken by sample_id
    assert cases["median"]["sample_id"] == "s1"  # lower median of 25,25,29,30,31
    assert select_visual_cases(ids, labels, scores) == select_visual_cases(ids, labels, scores)
    only_one = select_visual_cases(["a", "b"], ["0", "0"], [1.0, 2.0])  # no label-1: not invented
    assert [case["role"] for case in only_one] == ["fixed_label0", "median", "worst"]
    with pytest.raises(ValueError):
        select_visual_cases(ids, labels, [1.0, float("inf"), 3.0, 4.0, 5.0])


def test_column_naming_roundtrip() -> None:
    assert parse_col(q8_metrics.col("S2", "m1", "psnr_db")) == ("S2", "m1", "psnr_db")
    assert parse_col("ALL__mem__process_peak_rss_mb") == ("ALL", "mem", "process_peak_rss_mb")
    assert parse_col("sample_id") is None and parse_col("X__a__b") is None


def test_storage_breakdown_components_sum_to_container(tmp_path: Path) -> None:
    aad = {"study_instance_uid": "u", "sop_instance_uid": "s", "model_sha256": "ab" * 32, "sampler_version": "v"}
    z = torch.randn(1, 1, 16, 16)
    image = torch.rand(1, 1, 16, 16)
    with NonceRegistry(tmp_path / "n.sqlite3") as registry:
        m1 = seal_anonymous_image(image, 250.0, KEY, bytes(16), rounds=1, use_permutation=False,
                                  latent_shape=tuple(z.shape), latent_dtype="float32", aad=aad, registry=registry)
        m2 = seal_latent(z, KEY, bytes([1]) * 16, rounds=1, use_permutation=False, aad=aad, registry=registry)
    for blob, mode in ((m1, "M1"), (m2, "M2")):
        parts = container_storage_breakdown(blob)
        assert parts["mode"] == mode and parts["total_bytes"] == len(blob)
        components = {k: v for k, v in parts.items() if k.endswith("_bytes") and k != "total_bytes"}
        assert sum(components.values()) == len(blob)
        assert parts["nonce_bytes"] == 16 and parts["tag_bytes"] == 32 and parts["private_metadata_bytes"] > 0
        assert parts["encoding_overhead_bytes"] >= 0
    m2_parts = container_storage_breakdown(m2)
    assert m2_parts["latent_payload_bytes"] == 256 * 4 and m2_parts["image_payload_bytes"] == 0


# ----------------------------------------------------------------- LPIPS fail-closed / deferred
def test_lpips_status_is_unavailable_without_local_backbone(tmp_path: Path) -> None:
    status = lpips_status(hub_dir=tmp_path)
    assert status["available"] is False and "downloading is not performed" in status["reason"]
    (tmp_path / "alexnet-owt-deadbeef.pth").write_bytes(b"not the real weights")
    bad = lpips_status(hub_dir=tmp_path)
    assert bad["available"] is False and "does not match" in bad["reason"]


def test_build_lpips_metric_fails_closed_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(q8_metrics, "lpips_status", lambda hub_dir=None: {"available": False, "reason": "no weights"})
    with pytest.raises(LpipsUnavailable, match="no weights"):
        q8_metrics.build_lpips_metric()


# ----------------------------------------------------------------- legacy storage + wrong key helper
def test_legacy_codec_matches_predecessor_handoff() -> None:
    torch.manual_seed(0)
    image = torch.randn(1, 1, 256, 256) * 0.3 + 0.2
    payload, quantized = runner.encode_legacy(image)
    assert quantized.dtype == np.uint8 and payload.startswith(b"\x89PNG")
    assert torch.equal(runner.decode_legacy(payload, torch.device("cpu")), legacy_png_handoff(image))


def test_range_codec_is_lossless_to_8_bit_and_restores_range() -> None:
    image = torch.rand(1, 1, 256, 256) * 3 - 1
    payload, quantized = runner.encode_range(image)
    restored = runner.decode_range(payload, torch.device("cpu"))
    assert (restored - image).abs().max().item() <= 3.0 / 255 / 2 + 1e-6
    assert quantized.min() == 0 and quantized.max() == 255


def test_legacy_password_context_restores_constant() -> None:
    before = legacy.LEGACY_PASSWORD
    with runner.legacy_password("other"):
        assert legacy.LEGACY_PASSWORD == "other"
    assert legacy.LEGACY_PASSWORD == before
    z = torch.randn(1, 1, 256, 256)
    source = Path("past/SourceCode")
    encrypted = legacy.apply_legacy_key(z, source, "S1", "anonymize")
    with runner.legacy_password("Q8-wrong"):
        wrong = legacy.apply_legacy_key(encrypted, source, "S1", "deanonymize")
    assert torch.equal(legacy.apply_legacy_key(encrypted, source, "S1", "deanonymize"), z)
    assert (wrong - z).abs().max().item() > 1.0


# ----------------------------------------------------------------- runner (synthetic backend)
def run_synthetic(tmp_path: Path, name: str, role: str = "synthetic_selftest", limit: int = 3, extra=()) -> Path:
    argv = ["--evidence-role", role, "--synthetic-selftest", "--limit", str(limit),
            "--artifacts-root", str(tmp_path / name), *extra]
    args = runner.parse_args(argv)
    samples, backend, provenance = runner.build_synthetic_inputs(args)
    return runner.execute(args, samples, backend, provenance, io.StringIO(), io.StringIO())


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory) -> Path:
    return run_synthetic(tmp_path_factory.mktemp("q8"), "run", extra=("--storages", "range_preserving", "legacy"))


def read_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_runner_artifacts_validate_and_record_provenance(synthetic_run: Path) -> None:
    from scripts.artifact_schema import validate_run

    assert validate_run(synthetic_run)["valid"]
    manifest = json.loads((synthetic_run / "manifest.json").read_text())
    assert manifest["evidence_role"] == "synthetic_selftest" and manifest["formal"] is False
    assert manifest["synthetic_backend"] is True and manifest["lpips_computed_in_runner"] is False
    assert manifest["failures"] == 0 and manifest["raw_key_material_recorded"] is False
    assert manifest["main_storage"] == "range_preserving" and manifest["nonce_registry_count"] == 6
    assert set(manifest["arrays"]) == set(q8_metrics.ARRAY_KINDS)
    loaded = q8_metrics.load_verified_arrays(synthetic_run, json.loads((synthetic_run / "array_index.json").read_text()))
    assert loaded["recon_wrong_nonce"].shape == (3, 1, 256, 256) and loaded["x0"].shape == (3, 256, 256)
    # the public attacker view carries only 8-bit q values (no lo/hi)
    assert loaded["anonymous_public_q"].dtype == np.uint8


def test_runner_controls_and_stage_accounting(synthetic_run: Path) -> None:
    for row in read_rows(synthetic_run):
        assert row["S2__auth__wrong_key_rejected"] == row["S2__auth__wrong_nonce_rejected"] == "1"
        assert row["S2__auth__wrong_key_inverter_calls"] == row["S2__auth__wrong_nonce_inverter_calls"] == "0"
        assert row["S0__t1__reused_p_output"] == row["S1__t1__reused_p_output"] == "1"
        assert row["S2__t1__reused_p_output"] == "0"  # WHT in float32 is not bit-exact: real M2 path
        assert float(row["S2__m1lat__cosine"]) > 0.99
        # wrong key/nonce decrypt far worse than the legitimate path (plumbing sanity, not a gate)
        assert float(row["S2__wrongkey__psnr_db"]) < float(row["S2__m1__psnr_db"]) - 10
        assert float(row["S2__wrongnonce__psnr_db"]) < float(row["S2__m1__psnr_db"]) - 10
        assert "P__wrongkey__psnr_db" not in row and "S0__wrongnonce__psnr_db" not in row
        for stage in ("kdf_subkeys", "mac_compute", "mac_verify", "transform_inverse"):
            assert float(row[f"S2__runtime__micro_{stage}_seconds"]) >= 0
        e2e = sum(float(row[f"S2__runtime__{s}_seconds"]) for s in
                  ("container_open_total", "diffusion_invert", "diffusion_generate_recon"))
        assert float(row["S2__runtime__m1_to_reconstruction_seconds"]) == pytest.approx(e2e)
        anonymize = sum(float(row[f"S2__runtime__{s}_seconds"]) for s in
                        ("fixed_norm_scale", "transform_forward", "diffusion_generate_anonymous", "container_seal_total"))
        assert float(row["S2__runtime__cached_latent_to_m1_seconds"]) == pytest.approx(anonymize)
        assert not any("e2e" in key for key in row)
        assert not any(key.startswith("S2__mem__") for key in row)  # CPU run: GPU memory not fabricated
        assert int(row["S2__store__m1_total_bytes"]) > int(row["S2__store__m1_image_payload_bytes"])
        assert int(row["S0__store__m2_latent_payload_bytes"]) == 256 * 256 * 4


def test_runner_is_reproducible_across_runs(tmp_path: Path, synthetic_run: Path) -> None:
    other = run_synthetic(tmp_path, "again", extra=("--storages", "range_preserving", "legacy"))
    a, b = read_rows(synthetic_run), read_rows(other)
    deterministic = [k for k in a[0] if "__runtime__" not in k and "__mem__" not in k and "wall" not in k
                     and "peak_rss" not in k]
    assert [{k: row[k] for k in deterministic} for row in a] == [{k: row[k] for k in deterministic} for row in b]


def test_runner_fails_closed_without_per_sample_artifacts(tmp_path: Path, monkeypatch) -> None:
    original = runner.SyntheticBackend.generate
    calls = {"n": 0}

    def poisoned(self, latent):
        calls["n"] += 1
        out = original(self, latent)
        if calls["n"] > 15:  # poison from the second sample onward (about 11 calls per sample)
            out = out * float("nan")
        return out

    monkeypatch.setattr(runner.SyntheticBackend, "generate", poisoned)
    with pytest.raises(SystemExit) as caught:
        run_synthetic(tmp_path, "bad", limit=3, extra=("--warmup", "0"))
    assert caught.value.code == runner.FAILURE_EXIT_CODE
    run_dir = next((tmp_path / "bad" / "Q8_PILOT").iterdir())
    failures = json.loads((run_dir / "failures.json").read_text())
    assert failures[0]["sample_id"].startswith("synthetic_") and "non-finite" in failures[0]["error"]
    assert not (run_dir / "per_sample.csv").exists() and not (run_dir / "summary.json").exists()


@pytest.mark.parametrize("argv, message", [
    (["--evidence-role", "formal", "--synthetic-selftest"], "must be used together"),
    (["--evidence-role", "pilot"], "required for real runs"),
    (["--evidence-role", "synthetic_selftest", "--synthetic-selftest", "--storages", "legacy"], "--storages"),
])
def test_runner_rejects_inconsistent_requests(tmp_path: Path, argv, message) -> None:
    with pytest.raises(SystemExit, match=message):
        args = runner.parse_args(argv + ["--artifacts-root", str(tmp_path)])
        samples, backend, provenance = runner.build_synthetic_inputs(args)
        runner.execute(args, samples, backend, provenance, io.StringIO(), io.StringIO())


def test_formal_and_smoke_roles_are_bounded(tmp_path: Path) -> None:
    parsed = runner.parse_args(["--evidence-role", "formal", "--latent-cache", "x", "--checkpoint", "y",
                                "--split", "splits/security_v1.csv", "--limit", "10"])
    with pytest.raises(SystemExit, match="complete 200-image"):
        runner.validate_request(parsed, runner.synthetic_samples(10))
    smoke = runner.parse_args(["--evidence-role", "smoke", "--latent-cache", "x", "--checkpoint", "y"])
    with pytest.raises(SystemExit, match="limited to 4"):
        runner.validate_request(smoke, runner.synthetic_samples(5))
    with pytest.raises(SystemExit, match="at least 2"):
        runner.validate_request(smoke, runner.synthetic_samples(1))


# ----------------------------------------------------------------- analyzer
@pytest.fixture(scope="module")
def analysis(synthetic_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("q8out") / "a"
    manifest = q8_analyze.analyze(synthetic_run, out, allow_synthetic=True)
    return out, manifest


def table(out: Path, name: str) -> list[dict[str, str]]:
    with (out / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_all_deliverables_exist_with_declared_schema(analysis) -> None:
    out, manifest = analysis
    for name, fields in q8_analyze.FIELDS.items():
        with (out / name).open(newline="", encoding="utf-8") as handle:
            assert csv.DictReader(handle).fieldnames == fields
    assert (out / "fig_P8_visual_grid.png").read_bytes().startswith(b"\x89PNG")
    assert manifest["n_expected"] == manifest["n_ok"] == 3 and manifest["n_failed"] == 0
    assert manifest["synthetic_backend"] is True and manifest["lpips"]["state"] == "deferred"
    for name, digest in manifest["outputs_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_fidelity_has_ci_paired_differences_and_explicit_lpips_deferral(analysis) -> None:
    rows = table(analysis[0], "table_fidelity.csv")
    marginal = [r for r in rows if r["comparison"] == "marginal" and r["metric"] == "psnr_db" and r["path"] == "m1_image_inversion"
                and r["reference"] == "x0"]
    assert {(r["method"], r["storage"]) for r in marginal} == {(m, s) for m in q8_metrics.METHODS for s in ("range_preserving", "legacy")}
    assert all(float(r["ci95_low"]) <= float(r["mean"]) <= float(r["ci95_high"]) for r in marginal)
    assert {r["space"] for r in rows} == {"image", "latent"}
    assert any(r["comparison"] == "paired_difference" and r["baseline"] == "S0" and r["method"] == "S2" for r in rows)
    lpips = [r for r in rows if r["metric"] == "lpips"]
    assert lpips and all(r["n"] == "0" and r["mean"] == "" and r["status"].startswith("lpips deferred") for r in lpips)
    assert all(r["n_failed"] == "0" for r in rows if r["metric"] != "lpips")


def test_anonymity_has_rank_metrics_and_negative_controls(analysis) -> None:
    rows = table(analysis[0], "table_anonymity.csv")
    kinds = {r["kind"] for r in rows}
    assert kinds == {"image_difference", "nn_rank"}
    assert {r["method"] for r in rows if r["kind"] == "image_difference"} == set(q8_metrics.METHODS) | {"unrelated_control"}
    assert {r["view"] for r in rows if r["kind"] == "image_difference"} == {"public_available", "generator_float", "reference"}
    nn = [r for r in rows if r["kind"] == "nn_rank" and r["metric"] == "top1"]
    assert {r["control"] for r in nn} == {"true_match", "shuffled_truth_negative"}
    assert {r["scorer"] for r in nn} == {"cosine_raw", "pearson"}
    assert {r["view"] for r in nn} == {"public_available", "stored_pixels_q255", "generator_float"}
    p_true = [r for r in nn if r["method"] == "P" and r["control"] == "true_match" and r["scorer"] == "pearson"]
    assert all(float(r["mean"]) == 1.0 for r in p_true)  # positive control: unprotected P stays linkable
    p_shuffled = [r for r in nn if r["method"] == "P" and r["control"] == "shuffled_truth_negative"]
    assert all(float(r["mean"]) == 0.0 for r in p_shuffled)


def test_wrong_key_table_pairs_correct_wrong_and_controls(analysis) -> None:
    rows = table(analysis[0], "table_wrong_key.csv")
    s2 = {(r["condition"], r["path"]) for r in rows if r["method"] == "S2"}
    assert {("correct_key_nonce", "legitimate_open"), ("wrong_key", "bypass_auth_counterfactual"),
            ("wrong_nonce", "bypass_auth_counterfactual"), ("wrong_key", "authenticated_open"),
            ("wrong_nonce", "authenticated_open")} <= s2
    reject = [r for r in rows if r["metric"] == "rejection_rate" and r["method"] == "S2"]
    assert len(reject) == 2 and all(float(r["mean"]) == 1.0 and r["ci_method"] == "wilson 95%" for r in reject)
    calls = [r for r in rows if r["metric"] == "inverter_calls"]
    assert calls and all(float(r["max"]) == 0.0 for r in calls)
    assert any(r["method"] == "unrelated_control" for r in rows)
    na = [r for r in rows if r["method"] in {"P", "S0", "S1"} and r["metric"] == "rejection_rate"]
    assert len(na) == 3 and all(r["status"].startswith("not_applicable") for r in na)


def test_runtime_table_separates_stages_with_median_p95(analysis) -> None:
    rows = table(analysis[0], "table_runtime.csv")
    s2 = {r["stage"] for r in rows if r["method"] == "S2"}
    assert {"transform_forward", "micro_kdf_subkeys", "micro_mac_compute", "micro_mac_verify",
            "diffusion_invert", "diffusion_generate_recon", "container_seal_total", "container_open_total",
            "cached_latent_to_m1", "m1_to_reconstruction"} <= s2
    sums = [r for r in rows if r["stage"] in ("cached_latent_to_m1", "m1_to_reconstruction")]
    assert sums and all("EXCLUDES the initial x0->z inversion" in r["note"] for r in sums)
    assert not any("e2e" in r["stage"] for r in rows)
    timed = [r for r in rows if r["unit"] == "seconds" and r["n"] != "0"]
    assert all(r["median"] != "" and r["p95"] != "" and float(r["p95"]) >= float(r["median"]) for r in timed)
    kinds = {r["stage"]: r["kind"] for r in rows}
    assert kinds["micro_kdf_subkeys"] == "microbenchmark" and kinds["control_generate_wrong_key"] == "control"
    assert any(r["storage_cell"] == "legacy" for r in rows)
    assert all("GPU memory not measured" in r["status"] for r in timed)  # CPU run does not fabricate GPU peaks
    t1 = [r for r in rows if r["method"] == "S0" and r["stage"] == "diffusion_generate_t1"][0]
    assert t1["n"] == "0" and "no samples" in t1["status"]  # reused P output is excluded, not a zero-time sample
    assert any(r["stage"] == "process_peak_rss" and r["unit"] == "MiB" for r in rows)


def test_storage_table_splits_components_and_adds_up(analysis) -> None:
    rows = table(analysis[0], "table_storage.csv")

    def mean(method, mode, component):
        return float([r for r in rows if (r["method"], r["mode"], r["component"]) == (method, mode, component)][0]["mean"])

    parts = ("nonce_bytes", "tag_bytes", "manifest_bytes", "private_metadata_bytes", "latent_payload_bytes",
             "image_payload_bytes", "encoding_overhead_bytes")
    for mode in ("M1_image_only", "M2_latent_payload"):
        assert sum(mean("S2", mode, c) for c in parts) == pytest.approx(mean("S2", mode, "total_bytes"))
    assert mean("S2", "M1_image_only", "nonce_bytes") == 16 and mean("S2", "M1_image_only", "tag_bytes") == 32
    assert mean("S2", "M2_latent_payload", "latent_payload_bytes") == 262144
    assert mean("S2", "M2_plus_preview", "total_bytes") == pytest.approx(
        mean("S2", "M2_latent_payload", "total_bytes") + mean("S2", "M2_plus_preview", "preview_image_bytes"))
    assert mean("S0", "M1_image_only", "nonce_bytes") == 0
    assert [r for r in rows if r["method"] == "S0" and r["component"] == "nonce_bytes"][0]["applicable"] == "0"


def test_selection_is_preregistered_and_grid_is_deterministic(analysis, synthetic_run, tmp_path: Path) -> None:
    out, manifest = analysis
    selection = json.loads((out / "P8_selection.json").read_text())
    run = q8_analyze.Run(synthetic_run, allow_synthetic=True)
    scores = run.get("S2", "m1", "psnr_db")
    ranking = sorted(zip(run.sample_ids, scores), key=lambda item: (item[1], item[0]))
    roles = {case["role"]: case["sample_id"] for case in selection["cases"]}
    assert roles["worst"] == ranking[0][0] and roles["median"] == ranking[(len(ranking) - 1) // 2][0]
    assert selection["metric"] == q8_metrics.SELECTION_METRIC and selection["rule"] == q8_metrics.SELECTION_RULE
    again = tmp_path / "again"
    second = q8_analyze.analyze(synthetic_run, again, allow_synthetic=True)
    assert second["outputs_sha256"] == manifest["outputs_sha256"]  # byte-identical tables, grid, long rows


def test_long_rows_are_ci_ready(analysis) -> None:
    rows = table(analysis[0], "per_sample_long.csv")
    psnr = [r for r in rows if r["method"] == "S2" and r["group"] == "m1" and r["metric"] == "psnr_db"]
    assert len(psnr) == 3 and len({r["sample_id"] for r in psnr}) == 3 and all(r["patient_id"] for r in psnr)
    assert all(np.isfinite(float(r["value"])) for r in rows)


def test_analyzer_refuses_overwrite_synthetic_and_tampered_arrays(analysis, synthetic_run, tmp_path: Path) -> None:
    out, _ = analysis
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        q8_analyze.analyze(synthetic_run, out, allow_synthetic=True)
    with pytest.raises(SystemExit, match="allow-synthetic"):
        q8_analyze.analyze(synthetic_run, tmp_path / "x")
    assert not (tmp_path / "x").exists()


def test_analyzer_detects_array_and_failure_tampering(tmp_path: Path) -> None:
    run_dir = run_synthetic(tmp_path, "tamper")
    array_path = run_dir / "arrays" / "recon_m1.npy"
    data = bytearray(array_path.read_bytes()); data[-1] ^= 0xFF; array_path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        q8_analyze.analyze(run_dir, tmp_path / "o1", allow_synthetic=True)
    other = run_synthetic(tmp_path, "failed")
    (other / "failures.json").write_text("[]")
    with pytest.raises(SystemExit, match="failures"):
        q8_analyze.analyze(other, tmp_path / "o2", allow_synthetic=True)


def test_lpips_require_aborts_before_writing_and_injected_metric_is_used(synthetic_run, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(q8_metrics, "lpips_status", lambda hub_dir=None: {"available": False, "reason": "no weights"})
    monkeypatch.setattr(q8_analyze, "lpips_status", q8_metrics.lpips_status)
    with pytest.raises(SystemExit, match="LPIPS required but unavailable"):
        q8_analyze.analyze(synthetic_run, tmp_path / "req", lpips_mode="require", allow_synthetic=True)
    assert not (tmp_path / "req").exists()
    deferred = q8_analyze.analyze(synthetic_run, tmp_path / "auto", allow_synthetic=True)
    assert deferred["lpips"]["state"] == "deferred" and deferred["lpips"]["reason"] == "no weights"
    off = q8_analyze.analyze(synthetic_run, tmp_path / "off", lpips_mode="off", allow_synthetic=True)
    assert off["lpips"]["state"] == "not_requested"

    def fake_factory(_device):  # stands in for true LPIPS only to test the plumbing
        return lambda reference, other: float(np.abs(np.asarray(reference) - np.asarray(other)).mean())

    computed = q8_analyze.analyze(synthetic_run, tmp_path / "fake", metric_factory=fake_factory, allow_synthetic=True)
    assert computed["lpips"]["state"] == "computed"
    rows = table(tmp_path / "fake", "table_fidelity.csv")
    lpips = [r for r in rows if r["metric"] == "lpips" and r["comparison"] == "marginal"]
    assert lpips and all(r["status"] == "ok" and r["n"] == "3" and r["mean"] != "" for r in lpips)
    wrong = table(tmp_path / "fake", "table_wrong_key.csv")
    assert any(r["metric"] == "lpips" and r["condition"] == "wrong_key" and r["mean"] != "" for r in wrong)


# ======================================================================= review corrections
def tamper_per_sample(run_dir: Path, column: str, value: str) -> None:
    """Edit one CSV column AND keep summary.json consistent, so only the analyzer's own checks can object."""
    path = run_dir / "per_sample.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = list(reader.fieldnames), list(reader)
    for row in rows:
        row[column] = value
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if column in summary["numeric_means"]:
        summary["numeric_means"][column] = float(value)
        summary_path.write_text(json.dumps(summary))


def failure_of(tmp_path: Path, name: str) -> dict:
    run_dir = next((tmp_path / name / "Q8_PILOT").iterdir())
    assert not (run_dir / "per_sample.csv").exists() and not (run_dir / "summary.json").exists()
    return json.loads((run_dir / "failures.json").read_text())[0]


# ---- 1. authenticated wrong-key / wrong-nonce must be rejected with zero inverter calls
@pytest.mark.parametrize("scenario", ["wrong_key_accepted", "wrong_nonce_accepted", "rejected_after_inverter_call"])
def test_runner_aborts_if_authenticated_controls_are_not_fail_closed(tmp_path: Path, monkeypatch, scenario) -> None:
    real = runner.open_anonymous_image
    calls = {"n": 0}

    def sabotaged(blob, key, inverter, *, device="cpu"):
        calls["n"] += 1  # per sample: legitimate open, then wrong key, then forged nonce
        position = calls["n"] % 3 or 3
        if scenario == "wrong_key_accepted" and position == 2:
            return inverter(torch.zeros(1, 1, 256, 256))  # a broken verifier that lets the open proceed
        if scenario == "wrong_nonce_accepted" and position == 3:
            return inverter(torch.zeros(1, 1, 256, 256))
        if scenario == "rejected_after_inverter_call" and position == 2:
            inverter(torch.zeros(1, 1, 256, 256))
            raise runner.AuthenticationError("authentication failed")
        return real(blob, key, inverter, device=device)

    monkeypatch.setattr(runner, "open_anonymous_image", sabotaged)
    with pytest.raises(SystemExit) as caught:
        run_synthetic(tmp_path, "auth", limit=2, extra=("--warmup", "0"))
    assert caught.value.code == runner.FAILURE_EXIT_CODE
    failure = failure_of(tmp_path, "auth")
    assert "violated fail-closed rules" in failure["error"] and failure["sample_id"] == "synthetic_0000"


@pytest.mark.parametrize("column", ["S2__auth__wrong_key_rejected", "S2__auth__wrong_nonce_rejected"])
def test_analyzer_rejects_unrejected_authenticated_controls(tmp_path: Path, column: str) -> None:
    run_dir = run_synthetic(tmp_path, "evid")
    tamper_per_sample(run_dir, column, "0")
    with pytest.raises(SystemExit, match="inconsistent evidence: S2 authenticated"):
        q8_analyze.analyze(run_dir, tmp_path / "o", allow_synthetic=True)
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("column", ["S2__auth__wrong_key_inverter_calls", "S2__auth__wrong_nonce_inverter_calls"])
def test_analyzer_rejects_nonzero_inverter_calls(tmp_path: Path, column: str) -> None:
    run_dir = run_synthetic(tmp_path, "evid2")
    tamper_per_sample(run_dir, column, "1")
    with pytest.raises(SystemExit, match="zero inverter calls"):
        q8_analyze.analyze(run_dir, tmp_path / "o", allow_synthetic=True)


def test_analyzer_rejects_other_inconsistent_evidence(tmp_path: Path) -> None:
    run_dir = run_synthetic(tmp_path, "evid3")
    tamper_per_sample(run_dir, "x0_sha256", "0" * 64)
    with pytest.raises(SystemExit, match="x0_sha256"):
        q8_analyze.analyze(run_dir, tmp_path / "o", allow_synthetic=True)
    other = run_synthetic(tmp_path, "evid4")
    manifest_path = other / "manifest.json"
    manifest = json.loads(manifest_path.read_text()); manifest["nonce_registry_count"] = 5
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="nonce registry"):
        q8_analyze.analyze(other, tmp_path / "o2", allow_synthetic=True)


# ---- 2. E2.3 cache provenance, validated before any model exists
SPLIT_HASH, CKPT_HASH, SAMPLER_HASH = "a" * 64, "b" * 64, "c" * 64


def make_fake_cache(root: Path, count: int = 3, manifest_overrides=None, config_overrides=None, tamper_x0=False):
    from scripts.artifact_schema import write_synthetic_run
    samples = runner.synthetic_samples(count)
    rows = [{"sample_id": s.sample_id, "patient_id": s.patient_id, "label": s.label, "split_sha256": SPLIT_HASH,
             "local_path": "unused"} for s in samples]
    cache = root / "cache"
    write_synthetic_run(cache, [{"sample_id": s.sample_id, "metric": 1.0} for s in samples])
    latents = np.stack([s.z[0].numpy() for s in samples]).astype(np.float32)
    np.save(cache / "latents.npy", latents)
    manifest = json.loads((cache / "manifest.json").read_text())
    manifest.update({
        "task_id": "E2.3", "run_id": "E2.3_fake", "steps": 500, "pipeline": runner.LEGACY_PIPELINE,
        "guidance_scale": -1, "generation_conditioning": runner.GENERATION_CONDITIONING,
        "inversion_conditioning": "null=True (unconditional)", "source_sampler_sha256": SAMPLER_HASH,
        "checkpoint_hash": CKPT_HASH, "dataset_split_hash": SPLIT_HASH, "dtype": "float32",
        **(manifest_overrides or {})})
    (cache / "manifest.json").write_text(json.dumps(manifest))
    config = {"mode": "cache", "noise_level": 500, "pipeline": runner.LEGACY_PIPELINE, "guidance_scale": -1,
              "limit": 0, "split": "splits/fake.csv", **(config_overrides or {})}
    (cache / "config.json").write_text(json.dumps(config))
    with (cache / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "metric", "cache_index", "x0_sha256", "latent_sha256", "latent_finite"],
                                lineterminator="\n")
        writer.writeheader()
        for i, sample in enumerate(samples):
            writer.writerow({"sample_id": sample.sample_id, "metric": 1.0, "cache_index": i,
                             "x0_sha256": runner.sha256_array(sample.x0[0].numpy()),
                             "latent_sha256": runner.sha256_array(latents[i]), "latent_finite": 1})
    x0 = [sample.x0[0].numpy() for sample in samples]  # (1,256,256), the preprocess() shape
    if tamper_x0:
        x0[1] = x0[1] + np.float32(1e-3)
    return cache, rows, x0


def check_cache(cache, rows, x0, **overrides):
    kwargs = dict(noise_level=500, split_name="fake.csv", source_sampler_sha256=SAMPLER_HASH, checkpoint_hash=CKPT_HASH)
    kwargs.update(overrides)
    return runner.validate_latent_cache(cache, rows, x0, **kwargs)


def test_cache_provenance_accepts_exact_match_and_records_hashes(tmp_path: Path) -> None:
    cache, rows, x0 = make_fake_cache(tmp_path)
    result = check_cache(cache, rows, x0)
    hashes = result["hashes"]
    assert hashes["latent_cache_x0_verified"] == 3 and hashes["latent_cache_run_id"] == "E2.3_fake"
    for key in ("manifest", "config", "index"):
        assert hashes[f"latent_cache_{key}_sha256"] == hashlib.sha256(
            (cache / {"manifest": "manifest.json", "config": "config.json", "index": "per_sample.csv"}[key]).read_bytes()).hexdigest()
    assert result["latents"].shape == (3, 1, 256, 256)


@pytest.mark.parametrize("kind, kwargs, message", [
    ("manifest", {"steps": 250}, "steps=250"),
    ("manifest", {"pipeline": "other"}, "pipeline"),
    ("manifest", {"guidance_scale": 4}, "guidance_scale"),
    ("manifest", {"generation_conditioning": "x"}, "generation_conditioning"),
    ("manifest", {"inversion_conditioning": "x"}, "inversion_conditioning"),
    ("manifest", {"source_sampler_sha256": "d" * 64}, "source_sampler_sha256"),
    ("manifest", {"checkpoint_hash": "d" * 64}, "checkpoint_hash"),
    ("manifest", {"dataset_split_hash": "d" * 64}, "dataset_split_hash"),
    ("manifest", {"failure_reason": "oom"}, "failure_reason"),
    ("manifest", {"sample_ids": ["synthetic_0001", "synthetic_0000", "synthetic_0002"]}, "sample IDs"),
    ("config", {"noise_level": 250}, "noise_level"),
    ("config", {"limit": 5}, "limit"),
    ("config", {"mode": "smoke"}, "mode"),
    ("config", {"split": "splits/other.csv"}, "was built from split"),
])
def test_cache_provenance_fails_closed(tmp_path: Path, kind, kwargs, message) -> None:
    cache, rows, x0 = make_fake_cache(tmp_path, **{f"{kind}_overrides": kwargs})
    with pytest.raises(SystemExit, match=message):
        check_cache(cache, rows, x0)


def test_cache_provenance_checks_noise_level_checkpoint_and_x0(tmp_path: Path) -> None:
    cache, rows, x0 = make_fake_cache(tmp_path)
    with pytest.raises(SystemExit, match="steps=500"):
        check_cache(cache, rows, x0, noise_level=250)  # CLI noise level differs from the cache
    with pytest.raises(SystemExit, match="checkpoint_hash"):
        check_cache(cache, rows, x0, checkpoint_hash="e" * 64)
    with pytest.raises(SystemExit, match="shape"):
        check_cache(cache, rows, [array[0] for array in x0])  # (H,W) instead of the cache's (1,H,W)
    bad_cache, bad_rows, bad_x0 = make_fake_cache(tmp_path / "x", tamper_x0=True)
    with pytest.raises(SystemExit, match="x0 hash differs"):
        check_cache(bad_cache, bad_rows, bad_x0)


def test_cache_is_validated_before_the_model_is_created(tmp_path: Path, monkeypatch) -> None:
    import scripts.e1_ddim_runner as e1
    cache, rows, x0 = make_fake_cache(tmp_path, manifest_overrides={"steps": 250})
    split = tmp_path / "fake.csv"
    with split.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    checkpoint = tmp_path / "ckpt.pt"; checkpoint.write_bytes(b"x")
    source = tmp_path / "src" / "guided_diffusion"; source.mkdir(parents=True)
    (source / "gaussian_diffusion.py").write_text("# sampler")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(e1, "preprocess", lambda path: (torch.from_numpy(x0[0].copy()), None))
    monkeypatch.setattr(e1, "create_runtime", lambda *a, **k: pytest.fail("model created before cache validation"))
    args = runner.parse_args(["--evidence-role", "pilot", "--split", str(split), "--latent-cache", str(cache),
                              "--checkpoint", str(checkpoint), "--source-root", str(tmp_path / "src"), "--limit", "1"])
    with pytest.raises(SystemExit):
        runner.build_real_inputs(args)


# ---- 3. precise e2e names
def test_stage_sum_names_are_cached_latent_based() -> None:
    assert "e2e_anonymize" not in q8_metrics.STAGE_COMPONENT and "e2e_decrypt" not in q8_metrics.STAGE_COMPONENT
    assert q8_metrics.STAGE_COMPONENT["cached_latent_to_m1"] == "stage_sum"
    assert q8_metrics.STAGE_COMPONENT["m1_to_reconstruction"] == "stage_sum"


# ---- 4. unique patients
def test_runner_and_analyzer_require_unique_patients(tmp_path: Path) -> None:
    samples = runner.synthetic_samples(3)
    duplicated = [samples[0], runner.Sample("synthetic_x", samples[0].patient_id, "0", samples[1].x0, samples[1].z), samples[2]]
    args = runner.parse_args(["--evidence-role", "synthetic_selftest", "--synthetic-selftest", "--limit", "3"])
    with pytest.raises(SystemExit, match="patient IDs must be unique"):
        runner.validate_request(args, duplicated)
    run_dir = run_synthetic(tmp_path, "pat")
    tamper_per_sample(run_dir, "patient_id", "same_patient")
    with pytest.raises(SystemExit, match="patient IDs are not unique"):
        q8_analyze.analyze(run_dir, tmp_path / "o", allow_synthetic=True)


# ---- 5. public_available view
def test_public_available_view_is_stored_verified_and_asymmetric(synthetic_run: Path, analysis) -> None:
    arrays = q8_metrics.load_verified_arrays(synthetic_run, json.loads((synthetic_run / "array_index.json").read_text()))
    public, q8bit, anonymous = arrays["anonymous_public_available"], arrays["anonymous_public_q"], arrays["anonymous_float"]
    methods = q8_metrics.METHODS
    s2 = methods.index("S2")
    assert np.array_equal(public[:, s2], q8bit[:, s2].astype(np.float32) / np.float32(255))  # S2: q/255 only
    for method in ("P", "S0", "S1"):  # range-restored PNG: lo/hi public => original range recovered up to 8 bit
        k = methods.index(method)
        span = anonymous[:, k].max(axis=(1, 2)) - anonymous[:, k].min(axis=(1, 2))
        error = np.abs(public[:, k] - anonymous[:, k]).max(axis=(1, 2))
        assert (error <= span / 255 / 2 + 1e-5).all()
    rows = table(analysis[0], "table_anonymity.csv")
    image = [r for r in rows if r["kind"] == "image_difference" and r["view"] == "public_available"]
    assert {r["method"] for r in image} == set(methods)
    assert {r["metric"] for r in image} == {"psnr_db", "ssim", "lpips", "cosine", "linf"}
    assert all("asymmetric" in r["note"] for r in image)
    nn = [r for r in rows if r["kind"] == "nn_rank" and r["view"] == "public_available"]
    assert nn and all("asymmetric" in r["note"] for r in nn)
    assert any(r["view"] == "generator_float" and r["kind"] == "image_difference" for r in rows)  # diagnostic kept
    for row in read_rows(synthetic_run):
        assert all(f"{m}__anonpub__psnr_db" in row for m in methods)


# ---- 6. key / nonce policy by role
def test_key_policy_by_role() -> None:
    for role in ("synthetic_selftest", "smoke"):
        assert runner.resolve_key_mode(role, None) == "fixture"
        assert runner.resolve_key_mode(role, "random") == "random"
    for role in ("pilot", "formal"):
        assert runner.resolve_key_mode(role, None) == "random"
        with pytest.raises(SystemExit, match="limited to roles"):
            runner.resolve_key_mode(role, "fixture")
        with pytest.raises(SystemExit, match="limited to roles"):
            runner.parse_args(["--evidence-role", role, "--latent-cache", "x", "--checkpoint", "y", "--key-mode", "fixture"])


def test_random_key_policy_is_per_run_unique_and_never_recorded(tmp_path: Path, monkeypatch) -> None:
    captured = []

    class Spy(runner.KeyPolicy):
        def __init__(self, mode):
            super().__init__(mode)
            captured.append(self)

    monkeypatch.setattr(runner, "KeyPolicy", Spy)
    runs = []
    for name in ("one", "two"):
        args = runner.parse_args(["--evidence-role", "pilot", "--latent-cache", "x", "--checkpoint", "y",
                                  "--artifacts-root", str(tmp_path / name), "--warmup", "0"])
        samples, backend, provenance = runner.build_synthetic_inputs(runner.parse_args(
            ["--evidence-role", "synthetic_selftest", "--synthetic-selftest", "--limit", "3"]))
        runs.append(runner.execute(args, samples, backend, provenance, io.StringIO(), io.StringIO()))
    nonce_sets = []
    for run_dir, policy in zip(runs, captured):
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["key_mode"] == "random" and manifest["nonce_mode"] == runner.KEY_MODE_TEXT["random"]
        assert manifest["master_key_sha256"] == policy.master_key_sha256 and "public_fixture_key_sha256" not in manifest
        assert manifest["raw_key_material_recorded"] is False and manifest["nonce_registry_count"] == 6
        raw = policy.master_key
        for path in run_dir.rglob("*"):
            if path.is_file():
                data = path.read_bytes()
                assert raw not in data and raw.hex().encode() not in data, path
        import sqlite3
        with sqlite3.connect(run_dir / "nonce_registry.sqlite3") as db:
            nonce_sets.append({row[0] for row in db.execute("SELECT nonce FROM nonces")})
        assert len(nonce_sets[-1]) == 6
    assert captured[0].master_key != captured[1].master_key and captured[0].master_key != runner.FIXTURE_KEY
    assert not (nonce_sets[0] & nonce_sets[1])  # no cross-run nonce stream
    assert repr(captured[0]).count(captured[0].master_key.hex()) == 0


def test_fixture_mode_is_deterministic_and_recorded_as_such(synthetic_run: Path) -> None:
    manifest = json.loads((synthetic_run / "manifest.json").read_text())
    assert manifest["key_mode"] == "fixture" and manifest["nonce_mode"] == runner.KEY_MODE_TEXT["fixture"]
    assert manifest["public_fixture_key_sha256"] == hashlib.sha256(runner.FIXTURE_KEY).hexdigest()


def test_analyzer_rejects_fixture_keys_outside_smoke_and_selftest(tmp_path: Path) -> None:
    run_dir = run_synthetic(tmp_path, "keymode")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()); manifest["evidence_role"] = "pilot"; manifest["synthetic_backend"] = False
    manifest["raw_key_material_recorded"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="deterministic fixture key/nonces are not allowed for role pilot"):
        q8_analyze.analyze(run_dir, tmp_path / "o")


# ---- 7. LPIPS clipping disclosure
def test_clip_fractions_exact() -> None:
    array = np.zeros((2, 4, 4), dtype=np.float32)
    array[0, 0, :2] = [-0.5, 1.5]  # 2 of 16 pixels outside [0,1] in sample 0
    mean, maximum = q8_analyze.clip_fractions(array)
    assert maximum == pytest.approx(2 / 16) and mean == pytest.approx(1 / 16)


def test_lpips_rows_disclose_clipping_even_when_deferred(analysis, synthetic_run: Path) -> None:
    out, manifest = analysis
    assert manifest["lpips"]["input_view"] == q8_analyze.LPIPS_INPUT_VIEW and "UNCLIPPED" in manifest["lpips"]["input_view"]
    per_array = manifest["lpips"]["clip_fraction_per_array"]
    assert {"S2:m1", "S2:anonpub", "S0:wrongkey", "ALL:ctrl_unrelated", "P:anon"} <= set(per_array)
    arrays = q8_metrics.load_verified_arrays(synthetic_run, json.loads((synthetic_run / "array_index.json").read_text()))
    expected = q8_analyze.clip_fractions(arrays["recon_m1"][:, q8_metrics.METHODS.index("S2")])
    assert (per_array["S2:m1"]["mean"], per_array["S2:m1"]["max"]) == pytest.approx(expected)
    for name in ("table_fidelity.csv", "table_anonymity.csv", "table_wrong_key.csv"):
        lpips_rows = [r for r in table(out, name) if r["metric"] == "lpips"]
        assert lpips_rows
        for row in lpips_rows:
            assert "UNCLIPPED" in row["input_view"] and row["clip_fraction_mean"] != "" and row["clip_fraction_max"] != ""
        assert all(r["input_view"] == "" for r in table(out, name) if r["metric"] != "lpips")
    s2_row = [r for r in table(out, "table_fidelity.csv") if r["metric"] == "lpips" and r["method"] == "S2"
              and r["comparison"] == "marginal" and r["path"] == "m1_image_inversion" and r["storage"] == "range_preserving"][0]
    assert float(s2_row["clip_fraction_mean"]) == pytest.approx(expected[0])
