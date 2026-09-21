"""evaluation/validate_finetuned_generalization.py icin testler -- gercek
YOLO/OSNet/VisDrone verisi gerektiren detect+track akisi test edilmiyor
(projenin genel konvansiyonu, bkz. test_optimize_finetuned_tracker.py).
Burada SADECE saf mantik: 3-yollu sizinti kontrolu, agregasyon (naive
ortalama DEGIL), Pareto tespiti, ve kilit secimi -- ozellikle secimin
gercekten SADECE validasyon verisinden yapildigini, test sahnesi verisinin
KAZARA sizamayacagini dogrulayan testler."""
import math

import pytest

from evaluation.mot_metrics import evaluate_sequence
from evaluation.validate_finetuned_generalization import (
    BASELINE_PARAMS,
    CANDIDATES,
    TEST_SCENES,
    TRAIN_SCENES,
    VAL_SCENES,
    aggregate_rows,
    confirm_no_leakage_3way,
    is_pareto_optimal,
    select_locked_config,
)


# ---- Adim 1/17: 3-yollu sizinti kontrolu ----

def test_confirm_no_leakage_3way_passes_when_disjoint():
    assert confirm_no_leakage_3way(train=["a"], val=["b"], test=["c"]) is True


def test_confirm_no_leakage_3way_raises_on_train_val_overlap():
    with pytest.raises(SystemExit):
        confirm_no_leakage_3way(train=["a", "b"], val=["b", "c"], test=["d"])


def test_confirm_no_leakage_3way_raises_on_val_test_overlap():
    with pytest.raises(SystemExit):
        confirm_no_leakage_3way(train=["a"], val=["b", "c"], test=["c", "d"])


def test_confirm_no_leakage_3way_raises_on_train_test_overlap():
    with pytest.raises(SystemExit):
        confirm_no_leakage_3way(train=["a", "z"], val=["b"], test=["z", "d"])


# ---- Adim 2: split dogrulugu -- kullanicinin tam spesifikasyonuyla birebir ----

def test_split_scenes_match_specification():
    assert TRAIN_SCENES == ["uav0000086_00000_v", "uav0000339_00001_v"]
    assert VAL_SCENES == ["uav0000117_02622_v", "uav0000182_00000_v", "uav0000268_05773_v"]
    assert TEST_SCENES == ["uav0000137_00458_v", "uav0000305_00000_v"]


def test_production_split_has_no_overlap():
    # gercek modul-seviyesi sabitler uzerinde de calisir -- sentetik degil
    assert confirm_no_leakage_3way(TRAIN_SCENES, VAL_SCENES, TEST_SCENES) is True


# ---- Adim 3/4: aday konfigurasyon parse'i -- gercek repo varsayilanlarindan, TAHMIN DEGIL ----

def test_candidate_d_matches_repository_defaults():
    import inspect

    from evaluation.occlusion_event_forensics import BASELINE_CONF
    from tracking.adaptive_tracker import MAX_ASSOCIATION_DIST_RATIO, AdaptiveTracker

    sig = inspect.signature(AdaptiveTracker.__init__)
    d = CANDIDATES["D_finetuned_default"]
    assert d["track_thresh"] == sig.parameters["track_thresh"].default
    assert d["match_thresh"] == sig.parameters["match_thresh"].default
    assert d["assoc_dist_ratio"] == MAX_ASSOCIATION_DIST_RATIO
    assert d["conf"] == BASELINE_CONF


def test_baseline_params_use_same_tracker_defaults_as_candidate_d():
    d = CANDIDATES["D_finetuned_default"]
    assert BASELINE_PARAMS["assoc_dist_ratio"] == d["assoc_dist_ratio"]
    assert BASELINE_PARAMS["match_thresh"] == d["match_thresh"]
    assert BASELINE_PARAMS["track_thresh"] == d["track_thresh"]


def test_candidates_a_b_c_match_phase_4_2_report():
    # Faz 4.2'nin PHASE_4_2_REPORT.md'sindeki IDF1/MOTA/IDSW-optimum satirlariyla birebir
    assert CANDIDATES["A_idf1_optimum_4_2"] == dict(conf=0.55, assoc_dist_ratio=0.08, match_thresh=0.2, track_thresh=0.0)
    assert CANDIDATES["B_mota_optimum_4_2"] == dict(conf=0.60, assoc_dist_ratio=0.12, match_thresh=0.2, track_thresh=0.0)
    assert CANDIDATES["C_idsw_optimum_4_2"] == dict(conf=0.70, assoc_dist_ratio=0.08, match_thresh=0.3, track_thresh=0.0)


# ---- Adim 6: dogrulama agregasyonu -- toplama, YUZDE ORTALAMASI DEGIL ----

def test_aggregate_rows_sums_raw_tp_fp_fn_not_average_of_rates():
    trivial_gt = {1: [(1, (0, 0, 10, 10))]}
    acc = evaluate_sequence(trivial_gt, trivial_gt)
    rows = [{"tp": 8, "fp_det": 2, "fn_det": 0}, {"tp": 0, "fp_det": 0, "fn_det": 10}]

    agg = aggregate_rows(rows, {"s1": acc, "s2": acc})

    assert agg["tp"] == 8 and agg["fp_det"] == 2 and agg["fn_det"] == 10
    # global precision/recall global toplamdan -- (0.8+0.0)/2=0.4 GIBI naif bir ortalama DEGIL
    assert agg["precision"] == pytest.approx(0.8)
    assert agg["recall"] == pytest.approx(8 / 18)


def test_aggregate_rows_uses_motmetrics_overall_not_mean_of_scene_mota():
    # Sahne 1: 10 kare, tamami eslesti (MOTA=1.0). Sahne 2: 2 kare, tamami kayip (MOTA=0.0).
    # Naif ortalama: (1.0+0.0)/2=0.5. Doğru (sayim-tabanli) agregasyon: 1-(2/12)=0.8333.
    gt1 = {i: [(1, (0, 0, 10, 10))] for i in range(1, 11)}
    acc1 = evaluate_sequence(gt1, gt1)
    gt2 = {1: [(1, (0, 0, 10, 10))], 2: [(1, (0, 0, 10, 10))]}
    acc2 = evaluate_sequence(gt2, {})

    agg = aggregate_rows([{"tp": 0, "fp_det": 0, "fn_det": 0}] * 2, {"s1": acc1, "s2": acc2})

    assert agg["mota"] == pytest.approx(1 - 2 / 12)
    assert agg["mota"] != pytest.approx(0.5)


# ---- Adim 7: Pareto tespiti ----

def test_is_pareto_optimal_true_when_no_row_dominates_on_all_axes():
    a = {"idf1": 0.5, "mota": 0.1, "idsw": 10}
    b = {"idf1": 0.4, "mota": 0.2, "idsw": 20}  # a'dan iyi mota, kotu idf1/idsw -- birbirini domine etmiyor
    assert is_pareto_optimal(a, [a, b]) is True
    assert is_pareto_optimal(b, [a, b]) is True


def test_is_pareto_optimal_false_when_strictly_dominated():
    a = {"idf1": 0.5, "mota": 0.1, "idsw": 10}
    b = {"idf1": 0.6, "mota": 0.2, "idsw": 5}  # b, a'yi UC eksende de gecer
    assert is_pareto_optimal(a, [a, b]) is False
    assert is_pareto_optimal(b, [a, b]) is True


# ---- Adim 7/8/13: kilit secimi -- SADECE validasyondan, test sahnesi asla ----

def _row(candidate, idf1, mota, idsw, recall, split="validation", scene=None):
    return {"candidate": candidate, "split": split, "scene": scene,
            "idf1": idf1, "mota": mota, "idsw": idsw, "recall": recall}


def test_select_locked_config_picks_highest_idf1_excluding_baseline():
    rows = [
        _row("A_idf1_optimum_4_2", idf1=0.3, mota=0.1, idsw=50, recall=0.5),
        _row("B_mota_optimum_4_2", idf1=0.5, mota=0.0, idsw=60, recall=0.4),
        _row("E_baseline_vanilla", idf1=0.9, mota=0.9, idsw=1, recall=0.9),  # daha yuksek IDF1 ama uygun degil
    ]
    sel = select_locked_config(rows, test_scenes=["zzz_unrelated_scene"])
    assert sel["locked"]["candidate"] == "B_mota_optimum_4_2"


def test_select_locked_config_rejects_non_validation_split():
    rows = [_row("A", idf1=0.9, mota=0.9, idsw=1, recall=0.9, split="final_test")]
    with pytest.raises(SystemExit):
        select_locked_config(rows)


def test_select_locked_config_rejects_row_carrying_test_scene_data():
    rows = [_row("A", idf1=0.9, mota=0.9, idsw=1, recall=0.9, scene="uav0000137_00458_v")]
    with pytest.raises(SystemExit):
        select_locked_config(rows, test_scenes=["uav0000137_00458_v"])


def test_select_locked_config_reports_disagreement_via_separate_best_fields():
    rows = [
        _row("A", idf1=0.5, mota=0.0, idsw=50, recall=0.6),
        _row("B", idf1=0.4, mota=0.3, idsw=10, recall=0.5),
    ]
    sel = select_locked_config(rows, test_scenes=[])
    assert sel["best_idf1"]["candidate"] == "A"
    assert sel["best_mota"]["candidate"] == "B"
    assert sel["lowest_idsw"]["candidate"] == "B"
    assert sel["best_recall"]["candidate"] == "A"
