import pytest

from evaluation.mot_metrics import evaluate_sequence, load_mot_gt, summarize, xyxy_to_xywh


def test_xyxy_to_xywh_conversion():
    assert xyxy_to_xywh((10, 20, 30, 50)) == (10, 20, 20, 30)


def test_load_mot_gt_filters_non_pedestrian_and_low_visibility(tmp_path):
    gt_file = tmp_path / "gt.txt"
    gt_file.write_text(
        "1,1,10,10,20,40,1,1,0.9\n"  # pedestrian, gorunur -> dahil
        "1,2,50,50,20,40,1,7,0.9\n"  # class 7 (statik kisi) -> haric
        "1,3,90,90,20,40,1,1,0.05\n"  # cok dusuk gorunurluk -> haric
        "2,1,12,10,20,40,1,1,0.9\n"
    )
    by_frame = load_mot_gt(gt_file)
    assert set(by_frame[1]) == {(1, (10.0, 10.0, 20.0, 40.0))}
    assert set(by_frame[2]) == {(1, (12.0, 10.0, 20.0, 40.0))}


def test_perfect_tracking_gives_mota_and_idf1_of_one():
    gt = {
        1: [(100, (0, 0, 20, 40))],
        2: [(100, (5, 0, 20, 40))],
        3: [(100, (10, 0, 20, 40))],
    }
    hyp = {
        1: [(1, (0, 0, 20, 40))],
        2: [(1, (5, 0, 20, 40))],
        3: [(1, (10, 0, 20, 40))],
    }
    acc = evaluate_sequence(gt, hyp)
    summary = summarize({"perfect": acc})
    row = summary.loc["perfect"]
    assert row["mota"] == pytest.approx(1.0)
    assert row["idf1"] == pytest.approx(1.0)
    assert row["num_switches"] == 0


def test_id_switch_is_detected():
    gt = {
        1: [(100, (0, 0, 20, 40))],
        2: [(100, (5, 0, 20, 40))],
        3: [(100, (10, 0, 20, 40))],
    }
    hyp = {
        1: [(1, (0, 0, 20, 40))],
        2: [(1, (5, 0, 20, 40))],
        3: [(2, (10, 0, 20, 40))],  # ID degisti (1 -> 2), ayni kisi
    }
    acc = evaluate_sequence(gt, hyp)
    summary = summarize({"switched": acc})
    row = summary.loc["switched"]
    assert row["num_switches"] == 1
    assert row["idf1"] < 1.0


def test_missed_detection_counts_as_false_negative():
    gt = {1: [(100, (0, 0, 20, 40))], 2: [(100, (5, 0, 20, 40))]}
    hyp = {1: [(1, (0, 0, 20, 40))], 2: []}  # 2. karede tespit kayboldu
    acc = evaluate_sequence(gt, hyp)
    summary = summarize({"missed": acc})
    row = summary.loc["missed"]
    assert row["num_misses"] == 1
    assert row["mota"] < 1.0


def test_summarize_multiple_accumulators_includes_overall_row():
    gt = {1: [(100, (0, 0, 20, 40))]}
    hyp_good = {1: [(1, (0, 0, 20, 40))]}
    hyp_bad = {1: []}
    acc_good = evaluate_sequence(gt, hyp_good)
    acc_bad = evaluate_sequence(gt, hyp_bad)
    summary = summarize({"good": acc_good, "bad": acc_bad})
    assert "good" in summary.index
    assert "bad" in summary.index
    assert summary.loc["good", "mota"] > summary.loc["bad", "mota"]
