import numpy as np

from detection.yolo_detector import Detection
from evaluation.run_confidence_sweep import _PrecomputedExtractor, _run_tracker_at_threshold


def _det(cx, conf):
    return Detection((cx, 0.0, cx + 20.0, 40.0), conf, 0, "person")


def test_precomputed_extractor_returns_stored_embeddings_ignoring_args():
    embs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    extractor = _PrecomputedExtractor(embs)

    result = extractor.extract(frame=None, boxes_xyxy=[(0, 0, 1, 1), (2, 2, 3, 3)])

    assert result is embs  # aynen geri veriyor, frame/boxes argumanlarina hic bakmiyor


def test_run_tracker_at_threshold_drops_low_confidence_detections():
    # run_confidence_sweep'in butun mekanizmasi: tek bir dusuk-esik pass'inden
    # (detection, embedding) ciftlerini farkli conf_threshold'larla filtreleyip
    # HER SEFERINDE TAZE bir tracker'a beslemek. Gercek YOLO/OSNet gerekmeden
    # bu filtreleme+tracker mantigini dogrudan test eder.
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    per_frame = [
        [(_det(0, 0.9), np.array([1.0, 0.0])), (_det(50, 0.2), np.array([0.0, 1.0]))],
    ]

    high_thresh_hyp = _run_tracker_at_threshold(per_frame, conf_threshold=0.5, fps=30.0, dummy_frame=dummy_frame)
    low_thresh_hyp = _run_tracker_at_threshold(per_frame, conf_threshold=0.1, fps=30.0, dummy_frame=dummy_frame)

    assert len(high_thresh_hyp[1]) == 1  # sadece conf=0.9 tespiti track'e donustu
    assert len(low_thresh_hyp[1]) == 2   # ikisi de (0.9 ve 0.2) track'e donustu


def test_run_tracker_at_threshold_uses_fresh_tracker_state_per_call():
    # Iki ayri esik cagrisi birbirinin track ID sayacini/durumunu kirletmemeli
    # (her cagrida taze AdaptiveTracker olusturuluyor mu?).
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    per_frame = [[(_det(0, 0.9), np.array([1.0, 0.0]))]]

    first = _run_tracker_at_threshold(per_frame, conf_threshold=0.5, fps=30.0, dummy_frame=dummy_frame)
    second = _run_tracker_at_threshold(per_frame, conf_threshold=0.5, fps=30.0, dummy_frame=dummy_frame)

    assert first[1][0][0] == second[1][0][0]  # ayni girdiyle ayni track_id'den baslamali (taze durum)
