from detection.yolo_detector import Detection
from evaluation.occlusion_event_forensics import (
    _best_detection_at,
    _boundary_vis_range,
    _classify_side,
    _crowd_stats,
    classify_event,
)

BOX = (0.0, 0.0, 20.0, 40.0)


def test_classify_side_success_when_hyp_matched():
    assert _classify_side(vis=0.9, best_iou=0.0, hyp_match_exists=True) is None


def test_classify_side_marginal_visibility_is_E_regardless_of_detection():
    assert _classify_side(vis=0.25, best_iou=0.9, hyp_match_exists=False) == "E"


def test_classify_side_no_detection_at_all_is_A():
    assert _classify_side(vis=0.9, best_iou=0.0, hyp_match_exists=False) == "A"


def test_classify_side_partial_overlap_below_threshold_is_B():
    assert _classify_side(vis=0.9, best_iou=0.3, hyp_match_exists=False) == "B"


def test_classify_side_good_iou_but_stolen_is_C():
    assert _classify_side(vis=0.9, best_iou=0.7, hyp_match_exists=False) == "C"


def test_classify_event_recovered_when_both_sides_ok_and_same_hyp():
    assert classify_event(None, None, before_hyp=5, after_hyp=5) == "RECOVERED"


def test_classify_event_track_id_unstable_when_both_sides_ok_but_different_hyp():
    assert classify_event(None, None, before_hyp=5, after_hyp=9) == "D"


def test_classify_event_priority_A_over_B():
    # Ikisi de GERCEKTEN basarisiz (hyp'ler None) olmali ki reason'lar
    # anlamli olsun -- bkz. asagidaki "rescue" testi, bir taraf basariliysa
    # (hyp None degilse) o tarafin reason'u hic dikkate alinmamali.
    assert classify_event("B", "A", before_hyp=None, after_hyp=None) == "A"


def test_classify_event_priority_E_over_everything():
    assert classify_event("C", "E", before_hyp=None, after_hyp=None) == "E"


def test_classify_event_single_failing_side_reported():
    assert classify_event(None, "C", before_hyp=5, after_hyp=None) == "C"


def test_classify_event_search_window_rescue_is_not_mislabeled_by_exact_frame_reason():
    # GERCEK bug (ilk calistirmada bulundu, bkz. README "Faz 3.1"): tam sinir
    # karesinde gorunurluk dusuk oldugu icin reason_before="E" cikmis olsa
    # bile, compute_orr'un GENIS aramasi (search_window) o taraf icin GECERLI
    # bir hyp bulduysa (before_hyp/after_hyp None DEGILSE), event basarili
    # sayilmali -- "E" etiketiyle gizlenmemeli.
    assert classify_event(reason_before="E", reason_after=None, before_hyp=7, after_hyp=7) == "RECOVERED"
    assert classify_event(reason_before="E", reason_after="A", before_hyp=7, after_hyp=9) == "D"


def test_classify_event_unknown_on_inconsistent_input():
    # once/sonra "basarili" (reason=None) dendi ama hyp yine de None --
    # normalde olmamasi gereken bir tutarsizlik; sessizce yanlis bir
    # kategori uydurmak yerine acikca UNKNOWN dondurup isaretlemeli.
    assert classify_event(None, None, before_hyp=None, after_hyp=5) == "UNKNOWN"


def test_crowd_stats_counts_only_other_visible_identities():
    gt_by_frame_full = {
        1: [
            (1, BOX, 0.9),
            (2, (5.0, 0.0, 25.0, 40.0), 0.9),   # gorunur, yakinda -- sayilir
            (3, (100.0, 100.0, 120.0, 140.0), 0.9),  # gorunur ama uzak -- sayilir (crowd_count'a), IoU~0
            (4, (5.0, 0.0, 25.0, 40.0), 0.05),  # dusuk-vis -- SAYILMAZ
        ]
    }
    crowd_count, nearest_iou = _crowd_stats(gt_by_frame_full, frame=1, gt_id=1, box=BOX)
    assert crowd_count == 2  # id2 ve id3, id4 haric (dusuk vis), id1 kendisi haric
    assert nearest_iou > 0.0  # id2 ile ortusuyor


def test_crowd_stats_zero_when_alone_in_frame():
    gt_by_frame_full = {1: [(1, BOX, 0.9)]}
    crowd_count, nearest_iou = _crowd_stats(gt_by_frame_full, frame=1, gt_id=1, box=BOX)
    assert crowd_count == 0
    assert nearest_iou == 0.0


def test_best_detection_at_picks_highest_iou_and_its_confidence():
    detections_by_frame = {
        1: [
            Detection((0.0, 0.0, 20.0, 40.0), 0.42, 0, "person"),   # BOX ile IoU=1.0
            Detection((100.0, 100.0, 120.0, 140.0), 0.95, 0, "person"),  # alakasiz, yuksek conf ama IoU=0
        ]
    }
    best_iou, best_conf = _best_detection_at(detections_by_frame, frame=1, gt_box=BOX)
    assert best_iou == 1.0
    assert best_conf == 0.42  # IoU'ya gore secildi, confidence'a gore DEGIL


def test_best_detection_at_empty_frame_returns_zero():
    assert _best_detection_at({}, frame=1, gt_box=BOX) == (0.0, 0.0)


def test_boundary_vis_range_ignores_non_E_side_even_when_event_label_is_E():
    # GERCEK bug (MOT17-02 calistirmasinda bulundu, bkz. README): once
    # taraf "A" (vis=0.82, E'yle ILGISIZ), sonra taraf "E" (vis=0.21) --
    # event'in GENEL etiketi oncelik sirasina gore "E" olur (E > A), ama
    # 0.82 E'ye sebep olan bir deger DEGIL, dahil edilmemeli.
    rows = [
        {"vis_before": 0.82, "vis_after": 0.21, "reason_before": "A", "reason_after": "E", "label": "E"},
        {"vis_before": 0.22, "vis_after": 0.23, "reason_before": "E", "reason_after": "E", "label": "E"},
    ]
    lo, hi, n = _boundary_vis_range(rows)
    assert lo == 0.21 and hi == 0.23  # 0.82 ARALIGA GIRMEMELI
    assert n == 3  # sadece gercekten E'ye sebep olan 3 taraf (0.21, 0.22, 0.23)


def test_boundary_vis_range_none_when_no_E_reasons():
    rows = [{"vis_before": 0.9, "vis_after": 0.9, "reason_before": None, "reason_after": None, "label": "RECOVERED"}]
    assert _boundary_vis_range(rows) is None
