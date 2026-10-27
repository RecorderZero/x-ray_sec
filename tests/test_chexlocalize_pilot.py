import numpy as np
import pytest
import torch
from scripts.chexlocalize_pilot import normalize_map, select_balanced, evaluate_map


def test_constant_map_not_divided_by_zero():
    assert np.array_equal(normalize_map(np.ones((2, 2))), np.zeros((2, 2)))


def test_original_resolution_evaluation():
    result = evaluate_map(torch.zeros(1, 1, 2, 2), np.ones((4, 7), dtype=np.uint8))
    assert result['gt_positive_pixels'] == 28
    assert result['iou'] == 0 and result['raw_max'] == 0


def test_holdout_rejected():
    with pytest.raises(ValueError, match='held out'):
        select_balanced({'split':'test', 'mask_validation':'decoded_binary'}, 1)


def test_patient_unique_selection():
    records = [dict(sample_id='a', patient_id='p', mask_positive=True),
               dict(sample_id='b', patient_id='p', mask_positive=False),
               dict(sample_id='c', patient_id='q', mask_positive=False)]
    m = {'split':'val','mask_validation':'decoded_binary','records':records}
    assert [r['sample_id'] for r in select_balanced(m,1)] == ['a','c']
