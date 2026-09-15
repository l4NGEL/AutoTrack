"""evaluation/compare_finetune_vs_baseline.py icin testler -- gercek YOLO/
VisDrone verisi gerektiren _detection_recall/_tracking_metrics/main()
entegrasyon seviyesinde (test edilmiyor, projenin genel konvansiyonu).
Burada SADECE saf mantik: sizinti kontrolu ve GT-eslesme sayimi."""
import pytest

from evaluation.compare_finetune_vs_baseline import _count_gt_misses, confirm_no_leakage


def test_confirm_no_leakage_passes_when_disjoint():
    assert confirm_no_leakage(train_scenes=["a", "b"], eval_scenes=["c", "d"]) is True


def test_confirm_no_leakage_raises_on_overlap():
    with pytest.raises(SystemExit):
        confirm_no_leakage(train_scenes=["a", "b"], eval_scenes=["b", "c"])


def test_count_gt_misses_all_matched():
    # gt_rows kutulari (x,y,w,h) (load_mot_gt formati); det_boxes xyxy (YoloDetector formati)
    gt_rows = [(1, (0, 0, 10, 10)), (2, (100, 100, 10, 10))]
    det_boxes = [(0, 0, 10, 10), (100, 100, 110, 110)]
    total, missed = _count_gt_misses(gt_rows, det_boxes)
    assert (total, missed) == (2, 0)


def test_count_gt_misses_none_matched_when_no_detections():
    gt_rows = [(1, (0, 0, 10, 10)), (2, (100, 100, 10, 10))]
    total, missed = _count_gt_misses(gt_rows, [])
    assert (total, missed) == (2, 2)


def test_count_gt_misses_low_iou_counts_as_missed():
    # GT (0,0,10,10) ve tespit (8,8,18,18) -- cok az ortusuyor, IoU<0.5
    gt_rows = [(1, (0, 0, 10, 10))]
    det_boxes = [(8, 8, 18, 18)]
    total, missed = _count_gt_misses(gt_rows, det_boxes)
    assert (total, missed) == (1, 1)


def test_count_gt_misses_empty_gt_returns_zero_zero():
    total, missed = _count_gt_misses([], [(0, 0, 10, 10)])
    assert (total, missed) == (0, 0)
