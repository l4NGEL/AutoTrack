from evaluation.trace_track_lifecycle import _build_gt_hyp_timeline, _segments

BOX = (0.0, 0.0, 20.0, 40.0)


def test_segments_merges_consecutive_same_hyp_id():
    entries = [(1, 5), (2, 5), (3, 5), (4, 9), (5, 9), (6, None), (7, 5)]
    segs = _segments(entries)
    assert segs == [(5, 1, 3), (9, 4, 5), (None, 6, 6), (5, 7, 7)]


def test_segments_single_entry():
    assert _segments([(1, 5)]) == [(5, 1, 1)]


def test_segments_empty():
    assert _segments([]) == []


def test_build_gt_hyp_timeline_tracks_hyp_id_switch_across_a_gap():
    # gt_id=100 karelerde 1-3 hyp=1 ile eslesiyor, 4-6 gorunmuyor (dusuk vis,
    # timeline'a hic girmiyor), 7-9 tekrar gorunur ama hyp=2 (FARKLI ID).
    gt_by_frame_full = {
        1: [(100, BOX, 0.9)], 2: [(100, BOX, 0.9)], 3: [(100, BOX, 0.9)],
        4: [(100, BOX, 0.05)], 5: [(100, BOX, 0.05)], 6: [(100, BOX, 0.05)],
        7: [(100, BOX, 0.9)], 8: [(100, BOX, 0.9)], 9: [(100, BOX, 0.9)],
    }
    hyp_by_frame = {
        1: [(1, list(BOX))], 2: [(1, list(BOX))], 3: [(1, list(BOX))],
        7: [(2, list(BOX))], 8: [(2, list(BOX))], 9: [(2, list(BOX))],
    }
    timeline = _build_gt_hyp_timeline(gt_by_frame_full, hyp_by_frame)
    assert 100 in timeline
    frames_and_hyps = timeline[100]
    # sadece gorunur (vis>=0.2) kareler listelenir -- 4,5,6 YOK
    assert [f for f, _ in frames_and_hyps] == [1, 2, 3, 7, 8, 9]
    segs = _segments(frames_and_hyps)
    assert segs == [(1, 1, 3), (2, 7, 9)]  # ayni GT, iki farkli hyp_id -- gercek "switch"
