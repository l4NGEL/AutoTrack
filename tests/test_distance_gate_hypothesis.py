"""evaluation/test_distance_gate_hypothesis.py icin testler.

_collect_hyp / main() gercek YOLO+OSNet+MOT verisi gerektirir, entegrasyon
seviyesinde (bu dosyada test edilmiyor -- projenin genel konvansiyonu,
bkz. run_isolation_ablation.py'nin de birim testi yok). Burada SADECE bu
dosyanin kendi eklediği, saf mantik iceren fonksiyon test ediliyor:
_max_consecutive_jump.
"""
from evaluation.test_distance_gate_hypothesis import _max_consecutive_jump


def test_no_jump_when_track_stationary():
    hyp = {
        1: [(1, [100, 100, 120, 200])],
        2: [(1, [101, 100, 121, 200])],
        3: [(1, [102, 100, 122, 200])],
    }
    result = _max_consecutive_jump(hyp)
    assert result["jump_px"] < 5.0


def test_finds_large_consecutive_frame_jump():
    hyp = {
        1: [(1, [100, 100, 120, 200])],  # merkez (110, 150)
        2: [(1, [900, 100, 920, 200])],  # merkez (910, 150) -- 800px siçrama
    }
    result = _max_consecutive_jump(hyp)
    assert result["track_id"] == 1
    assert result["frame"] == 2
    assert result["jump_px"] == 800.0


def test_ignores_non_consecutive_frame_gap():
    """Track 20 kare kaybolup GERI GELDIGINDE (revival) bu ARDISIK bir kare
    cifti degildir -- Faz 2.4/2.6'nin taradigi hata sinifi bu degil (o,
    missed=0'dan missed=0'a TEK karede olan sicramaydi), bu yuzden buyuk bir
    mesafe farki burada YOK SAYILMALI."""
    hyp = {
        1: [(1, [100, 100, 120, 200])],
        30: [(1, [900, 100, 920, 200])],  # 29 kare sonra, ardisik degil
    }
    result = _max_consecutive_jump(hyp)
    assert result["jump_px"] == 0.0


def test_picks_largest_jump_across_multiple_tracks():
    hyp = {
        1: [(1, [0, 0, 10, 10]), (2, [0, 0, 10, 10])],
        2: [(1, [5, 0, 15, 10]), (2, [500, 0, 510, 10])],  # track1 kucuk, track2 buyuk siçrama
    }
    result = _max_consecutive_jump(hyp)
    assert result["track_id"] == 2
    assert result["jump_px"] == 500.0


def test_empty_input_returns_zero_jump():
    result = _max_consecutive_jump({})
    assert result["jump_px"] == 0.0
    assert result["track_id"] is None
