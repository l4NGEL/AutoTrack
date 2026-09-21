"""evaluation/optimize_finetuned_tracker.py icin testler -- gercek YOLO/
OSNet/VisDrone verisi gerektiren run_one_config/main() entegrasyon
seviyesinde (test edilmiyor, projenin genel konvansiyonu). Burada SADECE
saf mantik: Hungarian kutu eslesmesi, precision/recall/F1, sizinti kontrolu,
association parametre izgarasi."""
import math

import pytest

from evaluation.optimize_finetuned_tracker import _assoc_grid, confirm_no_leakage, match_boxes_hungarian, prf1


def test_match_boxes_all_matched():
    gt = [(0, 0, 10, 10), (100, 100, 110, 110)]
    det = [(0, 0, 10, 10), (100, 100, 110, 110)]
    tp, fp, fn = match_boxes_hungarian(gt, det)
    assert (tp, fp, fn) == (2, 0, 0)


def test_match_boxes_extra_detection_is_fp():
    gt = [(0, 0, 10, 10)]
    det = [(0, 0, 10, 10), (500, 500, 510, 510)]
    tp, fp, fn = match_boxes_hungarian(gt, det)
    assert (tp, fp, fn) == (1, 1, 0)


def test_match_boxes_missing_detection_is_fn():
    gt = [(0, 0, 10, 10), (500, 500, 510, 510)]
    det = [(0, 0, 10, 10)]
    tp, fp, fn = match_boxes_hungarian(gt, det)
    assert (tp, fp, fn) == (1, 0, 1)


def test_match_boxes_low_iou_does_not_count_as_match():
    gt = [(0, 0, 10, 10)]
    det = [(8, 8, 18, 18)]  # az ortusuyor, IoU<0.5
    tp, fp, fn = match_boxes_hungarian(gt, det)
    assert (tp, fp, fn) == (0, 1, 1)


def test_match_boxes_empty_gt_all_fp():
    tp, fp, fn = match_boxes_hungarian([], [(0, 0, 10, 10), (20, 20, 30, 30)])
    assert (tp, fp, fn) == (0, 2, 0)


def test_match_boxes_empty_det_all_fn():
    tp, fp, fn = match_boxes_hungarian([(0, 0, 10, 10), (20, 20, 30, 30)], [])
    assert (tp, fp, fn) == (0, 0, 2)


def test_match_boxes_prefers_best_global_assignment_not_greedy():
    # Faz 2.10'un ayni ruhunda: greedy yanlis secebilecegi bir senaryo
    gt = [(0, 0, 10, 10), (5, 5, 15, 15)]
    det = [(4, 4, 14, 14)]  # her iki GT ile de kismen ortusuyor, ama (5,5,15,15) ile daha iyi
    tp, fp, fn = match_boxes_hungarian(gt, det)
    assert tp + fn == 2
    assert tp <= 1  # tek tespit, en fazla 1 eslesme


def test_prf1_perfect():
    p, r, f1 = prf1(tp=10, fp=0, fn=0)
    assert p == 1.0 and r == 1.0 and f1 == 1.0


def test_prf1_zero_tp_returns_zero_not_nan_when_denominators_nonzero():
    p, r, f1 = prf1(tp=0, fp=5, fn=5)
    assert p == 0.0 and r == 0.0 and f1 == 0.0


def test_prf1_no_detections_no_gt_is_nan():
    p, r, f1 = prf1(tp=0, fp=0, fn=0)
    assert math.isnan(p) and math.isnan(r) and math.isnan(f1)


def test_confirm_no_leakage_passes_when_disjoint():
    assert confirm_no_leakage(train_scenes=["a", "b"], eval_scenes=["c", "d"]) is True


def test_confirm_no_leakage_raises_on_overlap():
    with pytest.raises(SystemExit):
        confirm_no_leakage(train_scenes=["a", "b"], eval_scenes=["b", "c"])


def test_assoc_grid_includes_current_value_and_stays_small():
    dist_ratios, match_threshes = _assoc_grid(0.08, 0.3)
    assert 0.08 in dist_ratios
    assert 0.3 in match_threshes
    assert len(dist_ratios) <= 4
    assert len(match_threshes) <= 4


def test_assoc_grid_match_thresh_never_negative():
    _, match_threshes = _assoc_grid(0.08, 0.05)  # -0.1 negatif olurdu
    assert all(m >= 0.0 for m in match_threshes)
