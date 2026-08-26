import numpy as np

from detection.yolo_detector import Detection
from tracking.adaptive_tracker import (
    AdaptiveTracker,
    Confidence,
    compute_adaptive_weights,
    DEFAULT_BASE_WEIGHTS,
)
from tracking.kalman_motion import KalmanBoxTracker
from tracking.adaptive_tracker import AdaptiveTrack


class StubExtractor:
    """Deterministic sahte embedding cikarici -- gercek OSNet agirligi
    yuklemeden AdaptiveTracker'in Re-ID kodu yolunu test eder."""

    def __init__(self, vectors_by_center):
        self.vectors_by_center = vectors_by_center

    def extract(self, frame, boxes_xyxy):
        out = []
        for box in boxes_xyxy:
            cx = round((box[0] + box[2]) / 2)
            out.append(self.vectors_by_center.get(cx, np.random.RandomState(0).randn(8)))
        arr = np.stack(out).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-8, None)


def _det(xyxy, conf=0.9, cls_name="person"):
    return Detection(xyxy=xyxy, conf=conf, cls_id=0, cls_name=cls_name)


def _frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_first_frame_creates_tracks_without_extractor():
    tracker = AdaptiveTracker()
    tracks = tracker.update([_det((0, 0, 20, 20)), _det((100, 100, 120, 120))], _frame())
    assert sorted(t.track_id for t in tracks) == [1, 2]
    assert all(t.embedding is None for t in tracks)


def test_track_id_persists_across_frames_without_reid():
    tracker = AdaptiveTracker()
    tracker.update([_det((0, 0, 20, 20))], _frame())
    tracks = tracker.update([_det((2, 2, 22, 22))], _frame())
    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_non_person_class_never_gets_reid_weight():
    track = AdaptiveTrack(
        track_id=1, xyxy=(0, 0, 20, 20), cls_name="car", conf=0.9,
        kalman=KalmanBoxTracker((0, 0, 20, 20)), ref_area=400.0,
    )
    alpha, beta, gamma = compute_adaptive_weights(track, 0, [(0, 0, 20, 20)], frame_diag=800.0)
    assert gamma == 0.0
    assert abs(alpha + beta - 1.0) < 1e-6


def test_occlusion_shifts_weight_from_iou_to_reid():
    track = AdaptiveTrack(
        track_id=1, xyxy=(0, 0, 20, 20), cls_name="person", conf=0.9,
        kalman=KalmanBoxTracker((0, 0, 20, 20)), ref_area=400.0,
    )
    no_overlap = compute_adaptive_weights(track, 0, [(0, 0, 20, 20), (500, 500, 520, 520)], frame_diag=800.0)
    heavy_overlap = compute_adaptive_weights(track, 0, [(0, 0, 20, 20), (2, 2, 22, 22)], frame_diag=800.0)
    assert heavy_overlap[2] > no_overlap[2]  # gamma (reid) up
    assert heavy_overlap[0] < no_overlap[0]  # alpha (iou) down


def test_fast_motion_shifts_weight_from_iou_to_motion():
    slow = AdaptiveTrack(
        track_id=1, xyxy=(0, 0, 20, 20), cls_name="person", conf=0.9,
        kalman=KalmanBoxTracker((0, 0, 20, 20)), ref_area=400.0,
    )
    fast = AdaptiveTrack(
        track_id=2, xyxy=(0, 0, 20, 20), cls_name="person", conf=0.9,
        kalman=KalmanBoxTracker((0, 0, 20, 20)), ref_area=400.0,
    )
    # Force the fast track's Kalman velocity state directly.
    fast.kalman.kf.x[4, 0] = 50.0
    fast.kalman.kf.x[5, 0] = 0.0

    slow_w = compute_adaptive_weights(slow, 0, [slow.xyxy], frame_diag=800.0)
    fast_w = compute_adaptive_weights(fast, 0, [fast.xyxy], frame_diag=800.0)
    assert fast_w[1] > slow_w[1]  # beta (motion) up
    assert fast_w[0] < slow_w[0]  # alpha (iou) down


def test_low_quality_crop_reduces_reid_weight():
    good = AdaptiveTrack(
        track_id=1, xyxy=(0, 0, 100, 100), cls_name="person", conf=0.95,
        kalman=KalmanBoxTracker((0, 0, 100, 100)), ref_area=10000.0,
    )
    poor = AdaptiveTrack(
        track_id=2, xyxy=(0, 0, 5, 5), cls_name="person", conf=0.3,
        kalman=KalmanBoxTracker((0, 0, 5, 5)), ref_area=10000.0,  # cok kucuk kirpim / referansa gore
    )
    good_w = compute_adaptive_weights(good, 0, [good.xyxy], frame_diag=800.0)
    poor_w = compute_adaptive_weights(poor, 0, [poor.xyxy], frame_diag=800.0)
    assert poor_w[2] < good_w[2]  # gamma (reid) down for low-quality crop


def test_reid_helps_resolve_identity_swap_after_crossing():
    # Iki kisi yatayda yaklasip capraz gecerken, salt IOU/hareket yerine
    # embedding onceligiyle dogru ID'leri tutabilmeli.
    tracker = AdaptiveTracker(weight_mode="adaptive")
    emb_a = np.zeros(8); emb_a[0] = 1.0
    emb_b = np.zeros(8); emb_b[1] = 1.0

    extractor = StubExtractor({10: emb_a, 200: emb_b})
    tracker.update([_det((0, 0, 20, 20)), _det((190, 0, 210, 20))], _frame(), extractor=extractor)

    # Bir sonraki karede A biraz saga, B biraz sola kaymis (yaklaşıyorlar) ama
    # embedding'ler sabit kalmis (ayni kisiler).
    extractor2 = StubExtractor({60: emb_a, 150: emb_b})
    tracks = tracker.update([_det((50, 0, 70, 20)), _det((140, 0, 160, 20))], _frame(), extractor=extractor2)

    by_id = {t.track_id: t for t in tracks}
    assert np.allclose(by_id[1].embedding[:2], [1.0, 0.0], atol=0.3)
    assert np.allclose(by_id[2].embedding[:2], [0.0, 1.0], atol=0.3)


def test_track_confidence_bucket_is_valid_enum():
    tracker = AdaptiveTracker()
    tracks = tracker.update([_det((0, 0, 20, 20), conf=0.9)], _frame())
    assert tracks[0].confidence_bucket in list(Confidence)
    assert 0.0 <= tracks[0].track_confidence <= 1.0


def test_track_pruned_after_buffer_exceeded():
    tracker = AdaptiveTracker(track_buffer=1)
    tracker.update([_det((0, 0, 20, 20))], _frame())
    tracker.update([], _frame())
    tracks = tracker.update([], _frame())
    assert tracks == []


def test_base_weights_sum_to_one():
    assert abs(sum(DEFAULT_BASE_WEIGHTS) - 1.0) < 1e-9
