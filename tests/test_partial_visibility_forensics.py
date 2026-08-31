from evaluation.partial_visibility_forensics import (
    classify_episode_outcome,
    classify_transition_reason,
    find_loss_transition_frame,
    find_partial_visibility_episodes,
)

BOX = (0.0, 0.0, 20.0, 40.0)


def _rows(gt_id, frames, vis):
    return {f: [(gt_id, BOX, vis)] for f in frames}


def _merge(*ds):
    out = {}
    for d in ds:
        for f, rows in d.items():
            out.setdefault(f, []).extend(rows)
    return out


def test_finds_episode_at_exactly_min_duration():
    gt = _rows(1, range(1, 61), 0.35)  # tam 60 kare, sinirda
    episodes = find_partial_visibility_episodes(gt, min_duration=60)
    assert len(episodes) == 1
    assert episodes[0] == {"gt_id": 1, "start": 1, "end": 60, "length": 60}


def test_rejects_episode_shorter_than_min_duration():
    gt = _rows(1, range(1, 60), 0.35)  # 59 kare, esigin altinda
    assert find_partial_visibility_episodes(gt, min_duration=60) == []


def test_episode_breaks_on_drop_below_low_threshold():
    gt = _merge(_rows(1, range(1, 40), 0.35), _rows(1, [40], 0.1), _rows(1, range(41, 80), 0.35))
    episodes = find_partial_visibility_episodes(gt, min_duration=30)
    # iki AYRI episode -- 0.1 karesi bandi kesiyor
    assert len(episodes) == 2
    assert episodes[0]["end"] == 39
    assert episodes[1]["start"] == 41


def test_episode_breaks_on_rise_above_high_threshold():
    gt = _merge(_rows(1, range(1, 40), 0.35), _rows(1, [40], 0.9), _rows(1, range(41, 80), 0.35))
    episodes = find_partial_visibility_episodes(gt, min_duration=30)
    assert len(episodes) == 2


def test_missing_gt_row_treated_as_outside_band():
    # 40. kare hic annotate edilmemis -- vis=0 kabul, bandi kesiyor (agir occlusion rejimi)
    gt = _merge(_rows(1, range(1, 40), 0.35), _rows(1, range(41, 80), 0.35))
    episodes = find_partial_visibility_episodes(gt, min_duration=30)
    assert len(episodes) == 2


def test_classify_episode_outcome_retained_no_gaps():
    seq = [(f, 7) for f in range(1, 61)]
    assert classify_episode_outcome(seq) == "RETAINED"


def test_classify_episode_outcome_recovered_same_id_after_gap():
    seq = [(f, 7) for f in range(1, 20)] + [(f, None) for f in range(20, 40)] + [(f, 7) for f in range(40, 61)]
    assert classify_episode_outcome(seq) == "RECOVERED"


def test_classify_episode_outcome_fragmented_different_id():
    # Faz 3.2'nin gercek deseni: 7 -> bosluk -> 40
    seq = [(f, 7) for f in range(1, 20)] + [(f, None) for f in range(20, 40)] + [(f, 40) for f in range(40, 61)]
    assert classify_episode_outcome(seq) == "FRAGMENTED"


def test_classify_episode_outcome_permanent_loss_all_none():
    seq = [(f, None) for f in range(1, 61)]
    assert classify_episode_outcome(seq) == "PERMANENT_LOSS"


def test_find_loss_transition_frame_after_last_match():
    seq = [(1, 7), (2, 7), (3, None), (4, None), (5, 40)]
    assert find_loss_transition_frame(seq) == 3  # 2'de son eslesme, 3 ilk bosluk


def test_find_loss_transition_frame_abrupt_switch_without_gap():
    # GERCEK bug: ilk versiyon sadece "eslesmeden BOSLUGA" gecisini ariyordu,
    # aralarinda hic None olmadan DOGRUDAN farkli bir ID'ye gecen (7->40)
    # durumu kacirip yanlislikla None donduruyordu.
    seq = [(1, 7), (2, 7), (3, 40), (4, 40)]
    assert find_loss_transition_frame(seq) == 3


def test_find_loss_transition_frame_permanent_loss_returns_first_frame():
    seq = [(10, None), (11, None), (12, None)]
    assert find_loss_transition_frame(seq) == 10


def test_classify_transition_reason_no_detection():
    assert classify_transition_reason(vis=0.35, best_iou=0.0, best_conf=0.0, hyp_match_exists=False) == "P1_NO_DETECTION"


def test_classify_transition_reason_low_iou():
    assert classify_transition_reason(vis=0.35, best_iou=0.3, best_conf=0.8, hyp_match_exists=False) == "P2_LOW_IOU"


def test_classify_transition_reason_track_thresh_gate():
    # IoU iyi ama guven tracker'in KENDI esiginin (0.5) altinda
    assert classify_transition_reason(vis=0.35, best_iou=0.7, best_conf=0.3, hyp_match_exists=False) == "P3_TRACK_THRESH_GATE"


def test_classify_transition_reason_assignment_conflict():
    # IoU iyi, guven de yeterli, ama yine de eslesmemis -- baska GT kapmis
    assert classify_transition_reason(vis=0.35, best_iou=0.7, best_conf=0.8, hyp_match_exists=False) == "P4_ASSIGNMENT_CONFLICT"


def test_classify_transition_reason_none_when_matched():
    assert classify_transition_reason(vis=0.35, best_iou=0.9, best_conf=0.9, hyp_match_exists=True) is None
