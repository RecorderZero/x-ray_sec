"""Frozen E2 linkage evaluator (AF-056).

Verbatim copies of the scorer / evaluation functions of ``scripts/t2bb_inner_linkage.py``
exactly as they were in the working tree when the formal E2 run executed
(source SHA-256 b02f78b6d10406da9dbf9523778c51b8acee64aaa4243bddec8e9205e14ba055, last
modified 2026-11-02 17:07 +0800, before the 2026-11-04 16:20 E2 run; snapshot kept at
paper_assets/share_s10/provenance/t2bb_inner_linkage_at_E2_b02f78b6.py.txt). The E2 runner
imports this committed module instead of the then-uncommitted helper, so a clean checkout
can replay E2. tests/security/test_share_linkage_eval_frozen.py checks the function
bodies against the snapshot.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

SEED = 1911
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
_DENSENET = None


def l2norm(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


@torch.no_grad()
def densenet_embed(arrays: list[np.ndarray], device: str) -> np.ndarray:
    global _DENSENET
    if _DENSENET is None:
        _DENSENET = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).eval().to(device)
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    out = []
    for a in arrays:
        lo, hi = float(a.min()), float(a.max())
        disp = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
        pil = Image.fromarray(np.rint(disp * 255).astype(np.uint8), mode="L")
        x = tf(pil).unsqueeze(0).to(device)
        f = F.relu(_DENSENET.features(x))
        out.append(F.adaptive_avg_pool2d(f, 1).flatten(1).cpu().numpy()[0])
    return np.asarray(out)


def similarity(arrays: list[np.ndarray], scorer: str, device: str) -> np.ndarray:
    flat = np.asarray([a.reshape(-1) for a in arrays], dtype=np.float64)
    if scorer == "cos":
        # raw SIGNED cosine. S0 (z_ano=D z) and S1 (z_ano=P D z) apply ONE
        # image-independent orthogonal map, so cos(z_ano_i,z_ano_j)=cos(z_i,z_j)
        # exactly -> this is the cipher's strongest linear transparency probe.
        return l2norm(flat) @ l2norm(flat).T
    if scorer == "abs":
        return l2norm(np.abs(flat)) @ l2norm(np.abs(flat)).T
    if scorer == "sorted_abs":
        s = np.sort(np.abs(flat), axis=1)
        return l2norm(s) @ l2norm(s).T
    if scorer == "norm":
        n = np.linalg.norm(flat, axis=1)
        return -np.abs(n[:, None] - n[None, :])        # closer norm -> higher score
    if scorer == "densenet121":
        e = l2norm(densenet_embed(arrays, device))
        return e @ e.T
    raise ValueError(scorer)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); avg = (csum - counts + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _boot_auc_ci(pos_by_q: list[np.ndarray], neg_by_q: list[np.ndarray],
                 boot: int = 2000, seed: int = SEED) -> tuple[float, float]:
    """Patient-level bootstrap 95% CI for the verification AUC: resample the query
    patients with replacement (each contributes its own pos/neg sims), recompute
    AUC, take the 2.5/97.5 percentiles.  Deterministic (seed=SEED)."""
    nq = len(pos_by_q)
    if nq < 2:
        return (float("nan"), float("nan"))
    brng = np.random.default_rng(seed)
    vals = np.empty(boot)
    idx_all = np.arange(nq)
    for b in range(boot):
        pick = brng.choice(idx_all, size=nq, replace=True)
        p = np.concatenate([pos_by_q[i] for i in pick])
        ng = np.concatenate([neg_by_q[i] for i in pick])
        vals[b] = auc(p, ng) if len(p) and len(ng) else np.nan
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return (round(float(lo), 4), round(float(hi), 4))


def evaluate(full_sim: np.ndarray, pid: list[str], roles: list[str], rng) -> dict:
    g = [i for i, r in enumerate(roles) if r == "gallery"]
    q = [i for i, r in enumerate(roles) if r == "query"]
    g_pid = [pid[i] for i in g]; q_pid = [pid[i] for i in q]
    sim = full_sim[np.ix_(q, g)]
    top1 = top5 = 0
    pos, neg = [], []
    pos_by_q, neg_by_q = [], []            # per-query, for patient-level bootstrap CI
    for qi, qp in enumerate(q_pid):
        order = np.argsort(-sim[qi])
        true_g = g_pid.index(qp)
        rank = int(np.where(order == true_g)[0][0])
        top1 += rank == 0; top5 += rank < 5
        qp_pos, qp_neg = [], []
        for gi, gp in enumerate(g_pid):
            (qp_pos if qp == gp else qp_neg).append(sim[qi, gi])
        pos.extend(qp_pos); neg.extend(qp_neg)
        pos_by_q.append(np.asarray(qp_pos)); neg_by_q.append(np.asarray(qp_neg))
    perm = rng.permutation(len(q_pid))
    nc = sum(int(np.argsort(-sim[qi])[0] == g_pid.index(q_pid[perm[qi]])) for qi in range(len(q_pid)))
    n = len(q_pid)
    ci_lo, ci_hi = _boot_auc_ci(pos_by_q, neg_by_q)
    return {"top1": top1 / n, "top5": top5 / n, "auc": auc(np.asarray(pos), np.asarray(neg)),
            "auc_ci_low": ci_lo, "auc_ci_high": ci_hi,
            "neg_control_top1": nc / n, "random_top1": 1.0 / len(g_pid)}

