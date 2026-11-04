"""AF-056: the committed frozen E2 evaluator equals the helper source used by the formal run."""

import ast
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "paper_assets/share_s10/provenance/t2bb_inner_linkage_at_E2_b02f78b6.py.txt"
FROZEN = ROOT / "scripts/share_linkage_eval.py"
FUNCS = ("l2norm", "densenet_embed", "similarity", "auc", "_boot_auc_ci", "evaluate")


def _functions(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS}


def test_snapshot_is_the_recorded_source() -> None:
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == \
        "b02f78b6d10406da9dbf9523778c51b8acee64aaa4243bddec8e9205e14ba055"


def test_frozen_functions_are_verbatim() -> None:
    snap, frozen = _functions(SNAPSHOT), _functions(FROZEN)
    assert set(frozen) == set(FUNCS)
    for name in FUNCS:
        assert frozen[name] == snap[name], name


def test_frozen_evaluator_has_ci_and_behaves() -> None:
    from scripts.share_linkage_eval import evaluate, similarity
    rng = np.random.default_rng(0)
    base = [rng.normal(size=(8, 8)).astype(np.float32) for _ in range(6)]
    arrays = base + [b + np.float32(0.01) * rng.normal(size=(8, 8)).astype(np.float32) for b in base]
    pid = [f"p{i}" for i in range(6)] * 2
    roles = ["gallery"] * 6 + ["query"] * 6
    res = evaluate(similarity(arrays, "cos", "cpu"), pid, roles, np.random.default_rng(1911))
    assert res["auc"] == 1.0 and res["top1"] == 1.0
    assert {"auc_ci_low", "auc_ci_high", "neg_control_top1", "random_top1"} <= set(res)
