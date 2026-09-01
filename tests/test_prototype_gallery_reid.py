import numpy as np

from evaluation.prototype_gallery_reid import GalleryAwareTracker
from reid.feature_extractor import cosine_similarity


class _StubTrack:
    def __init__(self, embedding=None, embedding_gallery=None):
        self.embedding = embedding
        self.embedding_gallery = embedding_gallery or []


def test_falls_back_to_base_score_when_gallery_empty():
    tracker = GalleryAwareTracker(use_embedding_gallery=True)
    t = _StubTrack(embedding=np.array([1.0, 0.0]), embedding_gallery=[])
    det = np.array([0.6, 0.8])
    expected_base = max(0.0, (cosine_similarity(t.embedding, det) + 1.0) / 2.0)
    assert tracker._reid_score(t, det) == expected_base


def test_gallery_match_better_than_ema_wins():
    tracker = GalleryAwareTracker(use_embedding_gallery=True)
    # EMA embedding (track.embedding) [1,0] vs det [0,1] -> cos=0 -> base skor 0.5
    # ama galeride det'e TAM eslesen [0,1] var -> cos=1 -> galeri skoru 1.0
    t = _StubTrack(embedding=np.array([1.0, 0.0]), embedding_gallery=[np.array([0.0, 1.0])])
    det = np.array([0.0, 1.0])
    assert tracker._reid_score(t, det) == 1.0


def test_ema_wins_when_gallery_match_is_worse():
    tracker = GalleryAwareTracker(use_embedding_gallery=True)
    # EMA [1,0] det [1,0] -> tam eslesme, base skor 1.0. Galeri [0,1] -> cos=0 -> 0.5.
    # max(1.0, 0.5) = 1.0 -- EMA hala kazanmali (galeri onu DUSURMEMELI).
    t = _StubTrack(embedding=np.array([1.0, 0.0]), embedding_gallery=[np.array([0.0, 1.0])])
    det = np.array([1.0, 0.0])
    assert tracker._reid_score(t, det) == 1.0


def test_gallery_ignored_when_use_embedding_gallery_is_false():
    tracker = GalleryAwareTracker(use_embedding_gallery=False)
    t = _StubTrack(embedding=np.array([1.0, 0.0]), embedding_gallery=[np.array([0.0, 1.0])])
    det = np.array([0.0, 1.0])
    # Galeri devre disi -- sadece base (EMA) skoru donmeli, galerinin
    # daha iyi eslesmesi (1.0) YOK SAYILMALI.
    assert tracker._reid_score(t, det) == 0.5


def test_none_det_embedding_returns_zero_even_with_gallery():
    tracker = GalleryAwareTracker(use_embedding_gallery=True)
    t = _StubTrack(embedding=np.array([1.0, 0.0]), embedding_gallery=[np.array([0.0, 1.0])])
    assert tracker._reid_score(t, None) == 0.0


def test_multiple_gallery_entries_picks_best_match():
    tracker = GalleryAwareTracker(use_embedding_gallery=True)
    t = _StubTrack(
        embedding=np.array([1.0, 0.0]),
        embedding_gallery=[np.array([0.0, 1.0]), np.array([-1.0, 0.0]), np.array([0.6, 0.8])],
    )
    det = np.array([0.6, 0.8])  # galerideki UCUNCU girdiyle TAM eslesiyor
    assert tracker._reid_score(t, det) == 1.0
