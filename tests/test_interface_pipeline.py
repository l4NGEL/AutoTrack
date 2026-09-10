"""interface/pipeline.py icin testler.

Alttaki YoloDetector/AdaptiveTracker/OSNetFeatureExtractor zaten kendi
test dosyalarinda dogrulanmis (bkz. test_adaptive_tracker.py,
test_reid_extractor.py, test_pipeline_smoke.py) -- burada SADECE bu
dosyanin kendi eklediği mantik test edilir: fps geri-dusme, sinif
filtresi, model onbellekleme (ayni parametrelerle ikinci TrackingPipeline
modeli yeniden YUKLEMEMELI), ve gercek kareler uzerinde uctan uca calisma.
"""
from pathlib import Path

import cv2
import pytest

from interface.pipeline import TrackingPipeline, read_source_fps

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_IMAGES = sorted(DATA_DIR.glob("diag_*.png"))


class _FakeCapture:
    def __init__(self, fps_value):
        self._fps_value = fps_value

    def get(self, _prop):
        return self._fps_value


@pytest.mark.parametrize("reported_fps", [0, None, 1, -5])
def test_read_source_fps_falls_back_to_30_when_unreliable(reported_fps):
    assert read_source_fps(_FakeCapture(reported_fps)) == 30.0


def test_read_source_fps_passes_through_valid_value():
    assert read_source_fps(_FakeCapture(24.0)) == 24.0


def test_classes_none_means_all_classes_and_normalizes_to_sorted_tuple():
    assert TrackingPipeline(classes=None).classes is None
    assert TrackingPipeline(classes=()).classes is None
    assert TrackingPipeline(classes=(7, 0, 2)).classes == (0, 2, 7)


def test_pipeline_uses_imagenet_backbone_when_checkpoint_is_none():
    pipeline = TrackingPipeline(reid_checkpoint=None, classes=(0,))
    assert pipeline.reid_checkpoint_missing is False  # bilerek None istendi, bu "eksik" sayilmaz
    assert pipeline.reid_extractor.has_trained_weights is False


def test_pipeline_falls_back_and_warns_when_checkpoint_path_does_not_exist():
    pipeline = TrackingPipeline(reid_checkpoint="checkpoints/bu_dosya_yok.pth.tar", classes=(0,))
    assert pipeline.reid_checkpoint_missing is True
    assert pipeline.reid_extractor.has_trained_weights is False


def test_detector_is_cached_across_pipelines_with_same_settings():
    """Ayni (agirlik, esik, sinif-filtresi) icin YOLO modeli DISKTEN
    TEKRAR yuklenmemeli -- arayuzde her yeni video/webcam oturumunda
    gozle gorulur bir gecikme olmamasinin sebebi bu."""
    p1 = TrackingPipeline(conf_threshold=0.35, classes=(0,))
    p2 = TrackingPipeline(conf_threshold=0.35, classes=(0,))
    assert p1.detector is p2.detector

    p3 = TrackingPipeline(conf_threshold=0.20, classes=(0,))
    assert p3.detector is not p1.detector  # farkli esik -> farkli detector

    p4 = TrackingPipeline(conf_threshold=0.35, classes=(0, 2))
    assert p4.detector is not p1.detector  # farkli sinif kumesi -> farkli detector


def test_new_pipeline_instances_have_independent_trackers():
    """Onbellek SADECE modelleri (YOLO/OSNet) paylasmali -- her kaynagin
    (her video/webcam oturumunun) KENDI track ID sayaci/kayip galerisi
    olmali, yoksa iki ayri video ayni ID uzayini paylasir."""
    p1 = TrackingPipeline(classes=(0,))
    p2 = TrackingPipeline(classes=(0,))
    assert p1.tracker is not p2.tracker


def test_unique_track_count_reflects_next_id_counter():
    pipeline = TrackingPipeline(classes=(0,))
    assert pipeline.unique_track_count == 0

    fake_det = type("Det", (), {"xyxy": (0, 0, 10, 10), "cls_name": "person", "conf": 0.9})()
    pipeline.tracker._new_track(fake_det, None)
    pipeline.tracker._new_track(fake_det, None)
    assert pipeline.unique_track_count == 2


@pytest.mark.skipif(not SAMPLE_IMAGES, reason="ornek data/diag_*.png bulunamadi")
def test_process_frame_runs_without_crashing_and_preserves_shape():
    pipeline = TrackingPipeline(classes=(0,))
    frame = cv2.imread(str(SAMPLE_IMAGES[0]))
    assert frame is not None

    annotated = pipeline.process_frame(frame.copy())
    assert annotated.shape == frame.shape
    assert pipeline.frame_count == 1
