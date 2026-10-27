import numpy as np
import pytest

from scripts.localization_metrics import mask_metrics, pixel_average_precision


def test_perfect():
    g = np.array([[1, 0], [0, 1]])
    m = mask_metrics(g, g, 0.5)
    assert m['iou'] == m['dice'] == m['pixel_ap'] == 1
    assert m['pointing_hit']


def test_empty_gt_not_artificial_perfect_score():
    m = mask_metrics(np.zeros((2, 2)), np.zeros((2, 2)), 0)
    assert m['iou'] is None and m['dice'] is None and m['pixel_ap'] is None
    assert m['pointing_hit'] is None and m['predicted_positive_fraction'] == 0


def test_tied_ap_is_prevalence():
    assert pixel_average_precision([0, 0, 0, 0], [0, 1, 0, 0]) == 0.25


def test_no_prediction_on_positive():
    m = mask_metrics(np.zeros((2, 2)), np.array([[0, 1], [0, 0]]), 0)
    assert m['iou'] == m['dice'] == 0 and m['pointing_hit'] is False


@pytest.mark.parametrize('h,g', [([[float('nan')]], [[1]]), ([[1]], [[2]]),
                               ([[1, 0]], [[1]]), ([], [])])
def test_reject_invalid(h, g):
    with pytest.raises(ValueError):
        mask_metrics(h, g, 0.5)
