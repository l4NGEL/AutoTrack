"""Gercek YOLO + AdaptiveTracker + OSNet ile uctan uca, ekran gerektirmeyen
duman testi. Repo'daki gercek kamera goruntuleri (data/diag_*.png) uzerinde
calisir; pipeline'in cokmeden ve mantikli degerler uretmesini dogrular."""
from pathlib import Path

import cv2
import pytest

from detection.yolo_detector import YoloDetector
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_IMAGES = sorted(DATA_DIR.glob("diag_*.png"))


@pytest.mark.skipif(not SAMPLE_IMAGES, reason="ornek data/diag_*.png bulunamadi")
def test_full_pipeline_runs_without_crashing_on_real_frames():
    detector = YoloDetector(weights="yolov8n.pt", conf_threshold=0.35)
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0")
    tracker = AdaptiveTracker()

    total_dets = 0
    for path in SAMPLE_IMAGES:
        frame = cv2.imread(str(path))
        assert frame is not None, f"okunamadi: {path}"

        detections = detector.detect(frame)
        total_dets += len(detections)
        tracks = tracker.update(detections, frame, extractor=extractor)

        for t in tracks:
            assert 0.0 <= t.track_confidence <= 1.0
            assert t.confidence_bucket is not None

    # Bu goruntuler bir masaustu kamera diyagnostigi (bkz. diagnose_camera.py)
    # ve hicbir insan icermiyor -- dogrulanan sey pipeline'in hic hata
    # vermeden N kareyi islemesi, Re-ID eslesme kalitesi DEGIL. OSNet'in
    # gercek kirpimlar uzerinde calistigi dogrulama tests/test_reid_extractor.py
    # (sentetik kirpim) ve evaluation/run_ablation.py (gercek MOT verisi) ile
    # yapilir. Bu sayi 0 ise extractor.extract() bu koşuda hic cagirilmamis
    # demektir -- bu durumu sessizce "basarili" gibi gostermemek icin yazdiriyoruz.
    print(f"\n[smoke] {len(SAMPLE_IMAGES)} kare islendi, toplam tespit: {total_dets}")
