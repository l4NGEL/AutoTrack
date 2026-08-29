import math

from evaluation.occlusion_recovery_rate import compute_orr, frame_level_match_rate, summarize_events

BOX = (0.0, 0.0, 20.0, 40.0)  # IoU=1.0 ile kendisiyle eslesir, hepsi ayni yerde durur


def _visible_frames(gt_id, frames, vis=0.9, box=BOX):
    return {f: [(gt_id, box, vis)] for f in frames}


def _merge(*frame_dicts):
    out = {}
    for d in frame_dicts:
        for f, rows in d.items():
            out.setdefault(f, []).extend(rows)
    return out


def _hyp(hyp_id, frames, box=BOX):
    return {f: [(hyp_id, box)] for f in frames}


def _merge_hyp(*frame_dicts):
    out = {}
    for d in frame_dicts:
        for f, rows in d.items():
            out.setdefault(f, []).extend(rows)
    return out


def test_recovered_event_when_same_hyp_id_before_and_after():
    gt = _visible_frames(100, range(1, 6))
    gt.update(_visible_frames(100, range(11, 16)))
    hyp = _hyp(1, list(range(1, 6)) + list(range(11, 16)))

    events = compute_orr(gt, hyp, min_occlusion_frames=5, max_recovery_gap=60)

    assert len(events) == 1
    e = events[0]
    assert e["gt_id"] == 100 and e["start"] == 6 and e["end"] == 10 and e["length"] == 5
    assert e["before_hyp"] == 1 and e["after_hyp"] == 1
    assert e["within_window"] is True


def test_id_switch_not_counted_as_recovered():
    gt = _visible_frames(100, range(1, 6))
    gt.update(_visible_frames(100, range(11, 16)))
    hyp = _merge_hyp(_hyp(1, range(1, 6)), _hyp(2, range(11, 16)))  # occlusion sonrasi FARKLI hyp id

    events = compute_orr(gt, hyp, min_occlusion_frames=5, max_recovery_gap=60)

    assert len(events) == 1
    assert events[0]["before_hyp"] == 1
    assert events[0]["after_hyp"] == 2
    summary = summarize_events(events, "test")
    assert summary["n_valid"] == 1
    assert summary["n_recovered"] == 0
    assert summary["orr"] == 0.0


def test_gap_shorter_than_min_occlusion_frames_ignored():
    gt = _visible_frames(100, range(1, 6))
    gt.update(_visible_frames(100, range(9, 14)))  # sadece 3 kare kopukluk (6,7,8)
    hyp = _hyp(1, list(range(1, 6)) + list(range(9, 14)))

    events = compute_orr(gt, hyp, min_occlusion_frames=5, max_recovery_gap=60)

    assert events == []


def test_gap_without_visible_frame_before_it_is_ignored():
    # gt_id 200 ilk goruldugu andan itibaren dusuk gorunurlukle basliyor --
    # "once gorunurdu, sonra kayboldu" degil, bu yuzden event SAYILMAMALI.
    gt = {f: [(200, BOX, 0.05)] for f in range(1, 6)}  # ilk 5 kare zaten dusuk-vis
    gt.update(_visible_frames(200, range(6, 11)))
    hyp = _hyp(1, range(6, 11))

    events = compute_orr(gt, hyp, min_occlusion_frames=5, max_recovery_gap=60)

    assert events == []


def test_within_and_beyond_window_split_by_max_recovery_gap():
    gt = _merge(
        _visible_frames(300, [1, 2, 3]),
        _visible_frames(300, [7, 8, 9]),  # ara: 4,5,6 -> uzunluk 3
        _visible_frames(301, [1, 2, 3]),
        _visible_frames(301, [6, 7, 8]),  # ara: 4,5 -> uzunluk 2
    )
    hyp = _merge_hyp(
        _hyp(1, [1, 2, 3, 7, 8, 9]),
        _hyp(2, [1, 2, 3, 6, 7, 8]),
    )

    events = compute_orr(gt, hyp, min_occlusion_frames=2, max_recovery_gap=2)

    by_id = {e["gt_id"]: e for e in events}
    assert by_id[300]["length"] == 3 and by_id[300]["within_window"] is False
    assert by_id[301]["length"] == 2 and by_id[301]["within_window"] is True


def test_search_window_falls_back_when_exact_boundary_frame_has_no_detection():
    # gap_start-1 (kare 5) icin hic hyp kutusu yok (detector o karede kacirdi),
    # ama kare 4'te var -- SEARCH_WINDOW geriye bakip onu bulmali.
    gt = _visible_frames(400, range(1, 6))
    gt.update(_visible_frames(400, range(11, 16)))
    hyp = _hyp(7, [1, 2, 3, 4])  # 5. karede KASITLI olarak yok
    hyp.update(_hyp(7, range(11, 16)))

    events = compute_orr(gt, hyp, min_occlusion_frames=5, max_recovery_gap=60)

    assert len(events) == 1
    assert events[0]["before_hyp"] == 7  # 4. kareden bulundu
    assert events[0]["after_hyp"] == 7


def test_no_hyp_anywhere_near_boundary_yields_invalid_event_not_a_crash():
    # once tarafinda SEARCH_WINDOW icinde HICBIR hyp kutusu yok.
    gt = _visible_frames(500, range(1, 6))
    gt.update(_visible_frames(500, range(11, 16)))
    hyp = _hyp(9, range(11, 16))  # sadece sonrasinda var, oncesinde hic yok

    events = compute_orr(gt, hyp, min_occlusion_frames=5, max_recovery_gap=60)

    assert len(events) == 1
    assert events[0]["before_hyp"] is None
    assert events[0]["after_hyp"] == 9
    summary = summarize_events(events, "test")
    assert summary["n_events"] == 1
    assert summary["n_valid"] == 0  # before_hyp None oldugu icin gecersiz
    assert math.isnan(summary["orr"])


def test_search_window_widening_recovers_event_missed_by_narrow_window():
    # gap_start-1 (kare 10) icin hic hyp yok, kare 4'te var (6 kare geri) --
    # dar pencere (5) bulamaz, genis pencere (6+) bulmali.
    gt = _visible_frames(600, range(1, 11))
    gt.update(_visible_frames(600, range(16, 21)))
    hyp = _hyp(3, [1, 2, 3, 4])  # 5-10 arasi hic yok (occlusion'a kadar bosluk)
    hyp.update(_hyp(3, range(16, 21)))

    narrow = compute_orr(gt, hyp, min_occlusion_frames=5, max_recovery_gap=60, search_window=5)
    wide = compute_orr(gt, hyp, min_occlusion_frames=5, max_recovery_gap=60, search_window=6)

    assert narrow[0]["before_hyp"] is None  # 10'dan 5 geri = kare5, hala bos
    assert wide[0]["before_hyp"] == 3  # 10'dan 6 geri = kare4, orada var


def test_frame_level_match_rate_counts_visible_matches_only():
    FAR = (500.0, 500.0, 520.0, 540.0)  # BOX'tan tamamen ayri konum (IoU=0 garantili)
    gt = {
        1: [(1, BOX, 0.9), (2, FAR, 0.05)],  # id2 dusuk-vis -> sayilmaz (konumu da ayri, greedy'yi etkilemesin diye)
        2: [(1, BOX, 0.9)],
    }
    hyp = {
        1: [(10, BOX)],   # id1 ile eslesir (IoU=1.0), id2'yle alakasiz
        2: [(10, FAR)],   # id1'in konumuyla alakasiz -> eslesmez
    }
    n_visible, n_matched, rate = frame_level_match_rate(gt, hyp)
    assert n_visible == 2  # sadece vis>=0.2 olan iki (kare,gt) cifti (id2 haric)
    assert n_matched == 1
    assert rate == 0.5


def test_frame_level_match_rate_nan_when_no_visible_gt():
    n_visible, n_matched, rate = frame_level_match_rate({}, {})
    assert n_visible == 0 and n_matched == 0
    assert math.isnan(rate)


def test_summarize_events_counts_and_orr_ratio():
    events = [
        {"before_hyp": 1, "after_hyp": 1},  # recovered
        {"before_hyp": 1, "after_hyp": 2},  # not recovered
        {"before_hyp": None, "after_hyp": 1},  # gecersiz
    ]
    summary = summarize_events(events, "ornek")
    assert summary["n_events"] == 3
    assert summary["n_valid"] == 2
    assert summary["n_recovered"] == 1
    assert summary["orr"] == 0.5
