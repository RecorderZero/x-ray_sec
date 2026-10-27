"""Explicit pixel-mask metrics; no threshold fitting on test data.

Metrics use the original annotation grid: resize the heatmap, not ground truth.
Empty-GT images have no defined lesion overlap/AP/pointing score and are reported
separately via predicted_positive_fraction. AP groups tied scores (not arbitrary
pixel ordering). These are generic metrics, not a claim of bit-identical
CheXlocalize official evaluation; use the official evaluator for its leaderboard.
"""
import numpy as np


def pixel_average_precision(scores, truth):
    scores = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=bool).ravel()
    if scores.shape != truth.shape or not np.isfinite(scores).all():
        raise ValueError("finite scores and equal shapes required")
    positives = int(truth.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    y, s = truth[order], scores[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.cumsum(y)[ends]
    precision = tp / (ends + 1)
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def mask_metrics(heatmap, mask, threshold):
    """Evaluate the supplied finite heatmap with a previously fixed threshold.

    Pointing: first row-major argmax (ties documented, no best-point selection).
    No normalization is performed here. Callers must record whether the input
    is raw or normalized and separately retain raw amplitude when normalizing.
    """
    h = np.asarray(heatmap, dtype=np.float64)
    g = np.asarray(mask)
    if h.ndim != 2 or h.size == 0 or h.shape != g.shape:
        raise ValueError("nonempty equal 2-D shapes required")
    if not np.isfinite(h).all() or not np.isfinite(threshold):
        raise ValueError("finite heatmap and threshold required")
    if not np.isin(g, [0, 1]).all():
        raise ValueError("mask must be binary, not a box or interpolated probabilities")
    g = g.astype(bool)
    p = h > threshold
    ng, npred = int(g.sum()), int(p.sum())
    intersection = int((p & g).sum())
    return {
        "gt_positive_pixels": ng,
        "predicted_positive_fraction": float(p.mean()),
        "iou": intersection / (ng + npred - intersection) if ng else None,
        "dice": 2 * intersection / (ng + npred) if ng else None,
        "pointing_hit": bool(g.ravel()[np.argmax(h)]) if ng else None,
        "pixel_ap": pixel_average_precision(h, g),
        "threshold": float(threshold),
        "flat_heatmap": bool(h.max() == h.min()),
    }
