import numpy as np

from tracking.bytetrack_wrapper import Track
from viz.overlay_renderer import draw_tracks


def _blank_frame():
    return np.zeros((200, 200, 3), dtype=np.uint8)


def test_draw_tracks_draws_box_for_currently_matched_track():
    frame = _blank_frame()
    track = Track(track_id=1, xyxy=(50, 50, 100, 100), cls_name="person", conf=0.9, missed=0)

    out = draw_tracks(frame, [track], priority_results={})

    region = out[50:100, 50:100]
    assert region.sum() > 0  # kutu cizgisi bu bolgede piksel degistirmis olmali


def test_draw_tracks_skips_box_for_missed_track():
    """2026-08-27: kullanicinin video uzerinde gozlemledigi 'burada kimse
    yokken kutu var' sorunu -- missed>0 (bu karede gercek tespit yok, sadece
    Kalman ekstrapolasyonu) track'ler artik cizilmiyor."""
    frame = _blank_frame()
    track = Track(track_id=1, xyxy=(50, 50, 100, 100), cls_name="person", conf=0.9, missed=1)

    out = draw_tracks(frame, [track], priority_results={})

    assert np.array_equal(out, frame)  # hic piksel degismedi -- kutu cizilmedi


def test_draw_tracks_mixed_list_only_draws_matched_ones():
    frame = _blank_frame()
    seen = Track(track_id=1, xyxy=(10, 10, 40, 40), cls_name="person", conf=0.9, missed=0)
    coasting = Track(track_id=2, xyxy=(120, 120, 160, 160), cls_name="person", conf=0.9, missed=3)

    out = draw_tracks(frame, [seen, coasting], priority_results={})

    assert out[10:40, 10:40].sum() > 0
    assert out[120:160, 120:160].sum() == 0
