"""Faz 4.3: DOGRULAMA/GENELLEME -- Faz 4.2'nin kesif niteligindeki sonuclarinin
gercek bir train/validation/test ayriminda tutup tutmadigini test eder.

**Neden bu script var:** Faz 4.2, parametre secimini (confidence + tracker
sweep) Faz 4.0'in ayni 2 "dokunulmamis" degerlendirme sahnesinde yapmisti --
kendi raporunun acikca belirttigi gibi bu bir DOGRULAMA degil, kesif
niteliginde bir hassasiyet analiziydi (parametre secimi ile "test" ayni
veriydi). Bu script bunu duzeltiyor: VisDrone-MOT'un 7 sahnelik tam
katalogundan (`data/visdrone_raw/samples.json`) 3 ayri sahne YENI bir
VALIDATION kumesi olarak kullaniliyor, parametre secimi SADECE bu kumede
yapiliyor, ve Faz 4.0/4.1/4.2'nin ayni 2 TEST sahnesi bu secim sirasinda
HIC incelenmiyor -- sadece secim kilitlendikten SONRA, tek seferlik bir
degerlendirme olarak kullaniliyor.

**Uc-yollu ayrim (asla degismez, `confirm_no_leakage_3way()` ile programatik
dogrulanir):**
  TRAIN : uav0000086_00000_v, uav0000339_00001_v      (Faz 4.1 fine-tune)
  VAL   : uav0000117_02622_v, uav0000182_00000_v, uav0000268_05773_v (YENI, Faz 4.3)
  TEST  : uav0000137_00458_v, uav0000305_00000_v      (Faz 4.0/4.1/4.2 ile AYNI)

**Yeniden kullanilan altyapi (SIFIR kopyalama):** `_PrecomputedExtractor`,
`_detect_and_embed_once`, `run_one_config`, `prf1` dogrudan `evaluation.
optimize_finetuned_tracker`'dan import edilir (Faz 4.2, kendisi de Faz 3.0'in
tek-gecis-coklu-filtre desenini yeniden kullanmisti). `run_one_config`'un
Faz 4.3 icin eklenen tek degisikligi -- `return_accumulator=True` -- Faz
4.2'nin hicbir cagrisini etkilemeyen, geriye-uyumlu bir eklemedir (varsayilan
False).

**Aday konfigurasyonlar -- kullanicinin kendi spesifikasyonundan, TAHMIN
EDILMEDI:** A/B/C, Faz 4.2'nin raporlanmis IDF1/MOTA/IDSW-optimum satirlari.
D ("fine-tuned varsayilan"), `occlusion_event_forensics.BASELINE_CONF=0.35`
+ `tracking/adaptive_tracker.AdaptiveTracker`'in KENDI constructor
varsayilanlaridir (track_thresh=0.5, match_thresh=0.3,
association_dist_ratio=MAX_ASSOCIATION_DIST_RATIO=0.08) -- dogrudan
kaynaktan okundu. E (baseline), ayni tracker varsayilanlari + `yolov8n.pt`.

**Dogrulama-agregasyonu (Adim 6):** ham TP/FP/FN sahneler arasinda
TOPLANIP (yuzdelerin ortalamasi ALINMADAN) `prf1()`'e verilir; MOTA/IDF1/
IDSW icin `mot_metrics.summarize()`'a 3 sahnenin HAM `mm.MOTAccumulator`'i
birlikte verilip `generate_overall=True` ile motmetrics'in KENDI istatistiksel
olarak doğru coklu-sekans agregasyonu kullanilir (Faz 2.7/2.8'in 7-sekans
agregatiyla AYNI yontem).

**Kilit disiplini (Adim 7/8/13):** `select_locked_config()` SADECE
`split="validation"` etiketli satirlari kabul eder ve her satirin `scene`
degerinin TEST_SCENES disinda oldugunu ACIKCA dogrular (aksi halde
SystemExit) -- test sahnelerinin secime sizmasina karsi calisma-zamani
korumasi, sadece disiplin/niyet degil. Secim TEK SEFER yapilir, IDF1
(kullanicinin belirttigi PRIMARY metrik) ile eslesen en yuksek adayi secer;
agirlikli bir skor UYDURULMAZ (kullanicinin acik kurali).

Kullanim:
    python -m evaluation.validate_finetuned_generalization --checkpoint checkpoints/yolo_visdrone_finetune/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from detection.yolo_detector import YoloDetector
from evaluation.mot_metrics import load_mot_gt, read_seq_fps, summarize, xyxy_to_xywh
from evaluation.optimize_finetuned_tracker import (
    _detect_and_embed_once,
    prf1,
    run_one_config,
)
from reid.feature_extractor import OSNetFeatureExtractor

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "VisDrone"
REID_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "osnet_mot_finetuned" / "model.pth.tar"
DEFAULT_FINETUNED_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "yolo_visdrone_finetune" / "weights" / "best.pt"
BASELINE_WEIGHTS = "yolov8n.pt"

TRAIN_SCENES = ["uav0000086_00000_v", "uav0000339_00001_v"]
VAL_SCENES = ["uav0000117_02622_v", "uav0000182_00000_v", "uav0000268_05773_v"]
TEST_SCENES = ["uav0000137_00458_v", "uav0000305_00000_v"]  # Faz 4.0/4.1/4.2 ile AYNI -- degismedi

THRESHOLD_SWEEP_CONFS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
# Faz 4.1/Faz 4.2 disiplini: sessiz ikinci esigi (track_thresh varsayilani 0.5)
# devre disi birakip sweep'in dusuk conf degerlerini gercekten test etmesini saglar.
SWEEP_ASSOC_DIST_RATIO = 0.08
SWEEP_MATCH_THRESH = 0.3
SWEEP_TRACK_THRESH = 0.0

# Faz 4.2'nin raporlanmis Pareto adaylari (kullanicinin spesifikasyonundan, degistirilmedi)
CANDIDATES = {
    "A_idf1_optimum_4_2": dict(conf=0.55, assoc_dist_ratio=0.08, match_thresh=0.2, track_thresh=0.0),
    "B_mota_optimum_4_2": dict(conf=0.60, assoc_dist_ratio=0.12, match_thresh=0.2, track_thresh=0.0),
    "C_idsw_optimum_4_2": dict(conf=0.70, assoc_dist_ratio=0.08, match_thresh=0.3, track_thresh=0.0),
    # AdaptiveTracker.__init__ varsayilanlari + occlusion_event_forensics.BASELINE_CONF -- TAHMIN DEGIL.
    "D_finetuned_default": dict(conf=0.35, assoc_dist_ratio=0.08, match_thresh=0.3, track_thresh=0.5),
}
BASELINE_PARAMS = dict(conf=0.35, assoc_dist_ratio=0.08, match_thresh=0.3, track_thresh=0.5)
# LOCKED secimi icin uygun aday havuzu: A-D + esik sweep noktalari.
# E (baseline) HARIC -- o fine-tune edilmis model icin bir parametre adayi degil, referans/taban.
SELECTION_POOL_EXCLUDES_BASELINE = True


def confirm_no_leakage_3way(train=TRAIN_SCENES, val=VAL_SCENES, test=TEST_SCENES) -> bool:
    """Uc kumenin HICBIRININ birbiriyle kesismedigini programatik dogrular.
    Herhangi bir kesisim varsa (egitim<->val, egitim<->test, val<->test)
    SystemExit ile YUKSEK SESLE durur -- sessizce devam etmez."""
    pairs = [
        ("train", "val", set(train) & set(val)),
        ("train", "test", set(train) & set(test)),
        ("val", "test", set(val) & set(test)),
    ]
    overlaps = {f"{a}<->{b}": sorted(ov) for a, b, ov in pairs if ov}
    if overlaps:
        raise SystemExit(f"SIZINTI: {overlaps}")
    print(f"[sizinti kontrolu] train={train}\n  val={val}\n  test={test}\n  -- ucu arasinda kesisim YOK")
    return True


def aggregate_rows(rows: list, accumulators_by_scene: dict) -> dict:
    """Bir konfigurasyonun BIRDEN FAZLA sahnedeki satirlarini dogru sekilde
    agregeler: tespit P/R/F1 icin HAM tp/fp/fn TOPLANIR (yuzdelerin ortalamasi
    ALINMAZ), MOTA/IDF1/IDSW/MT/PT/ML icin motmetrics'in KENDI coklu-accumulator
    `generate_overall` mekanizmasi kullanilir (`accumulators_by_scene`: {sahne:
    mm.MOTAccumulator})."""
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp_det"] for r in rows)
    fn = sum(r["fn_det"] for r in rows)
    precision, recall, f1 = prf1(tp, fp, fn)

    summary = summarize(accumulators_by_scene)
    overall = summary.loc["OVERALL"] if len(accumulators_by_scene) > 1 else summary.iloc[0]

    return {
        "tp": tp, "fp_det": fp, "fn_det": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mota": float(overall["mota"]), "idf1": float(overall["idf1"]),
        "idsw": int(overall["num_switches"]),
        "fp_trk": int(overall["num_false_positives"]), "fn_trk": int(overall["num_misses"]),
        "mt": int(overall["mostly_tracked"]), "pt": int(overall["partially_tracked"]),
        "ml": int(overall["mostly_lost"]), "gt_obj": int(overall["num_objects"]),
    }


def is_pareto_optimal(candidate: dict, others: list) -> bool:
    """candidate, `others` icindeki HERHANGI bir satir tarafindan uc eksende
    BASKIN DEGILSE (idf1 yuksek, mota yuksek, idsw dusuk -- hepsi ayni anda >=,
    en az birinde >) Pareto-optimal sayilir."""
    for o in others:
        if o is candidate:
            continue
        dominates = (
            o["idf1"] >= candidate["idf1"] and o["mota"] >= candidate["mota"] and o["idsw"] <= candidate["idsw"]
            and (o["idf1"] > candidate["idf1"] or o["mota"] > candidate["mota"] or o["idsw"] < candidate["idsw"])
        )
        if dominates:
            return False
    return True


def select_locked_config(validation_aggregate_rows: list, test_scenes=TEST_SCENES) -> dict:
    """SADECE dogrulama-agregasyonu satirlarindan secim yapar. Her satirin
    `split=="validation"` oldugunu VE hicbir satirin test sahnesi verisi
    icermedigini ACIKCA dogrular -- test sahnelerinin secime kazara sizmasina
    karsi calisma-zamani korumasi (sadece disiplin degil)."""
    for r in validation_aggregate_rows:
        if r.get("split") != "validation":
            raise SystemExit(f"KURAL IHLALI: secim icin validation-disi satir verildi: {r.get('candidate')}")
        if r.get("scene") is not None and r["scene"] in test_scenes:
            raise SystemExit(f"KURAL IHLALI: secim TEST sahnesi verisi iceriyor: {r['scene']}")

    pool = [r for r in validation_aggregate_rows if not r["candidate"].startswith("E_baseline")]
    best_idf1 = max(pool, key=lambda r: r["idf1"])
    best_mota = max(pool, key=lambda r: r["mota"])
    lowest_idsw = min(pool, key=lambda r: r["idsw"])
    best_recall = max(pool, key=lambda r: r["recall"])
    pareto = [r for r in pool if is_pareto_optimal(r, pool)]

    return {
        "best_idf1": best_idf1, "best_mota": best_mota,
        "lowest_idsw": lowest_idsw, "best_recall": best_recall,
        "pareto_set": pareto,
        "locked": best_idf1,  # PRIMARY=IDF1 (kullanicinin spesifikasyonu) -- agirlikli skor UYDURULMADI
    }


def _run_candidates_on_scene(per_frame, gt_by_frame, fps, dummy_frame, candidates: dict,
                              scene: str, split: str, checkpoint: str, detection_fps: float) -> tuple:
    """Bir sahnede bir aday konfigurasyon kumesini calistirir. Doner:
    (satir_listesi, {aday_adi: mm.MOTAccumulator})."""
    rows, accs = [], {}
    for name, params in candidates.items():
        row, acc = run_one_config(per_frame, gt_by_frame, fps, dummy_frame=dummy_frame,
                                   return_accumulator=True, **params)
        row.update(scene=scene, candidate=name, split=split, checkpoint=checkpoint, detection_fps=detection_fps)
        rows.append(row)
        accs[name] = acc
    return rows, accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_FINETUNED_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "experiments")
    parser.add_argument("--limit-frames", type=int, default=None, help="Hizli duman testi icin")
    parser.add_argument("--skip-threshold-sweep", action="store_true")
    args = parser.parse_args()

    confirm_no_leakage_3way()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_str = str(args.checkpoint)

    reid_ckpt = str(REID_CHECKPOINT) if REID_CHECKPOINT.is_file() else None
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=reid_ckpt)

    # ================= ADIM 5/6: VALIDASYON =================
    val_rows: list = []
    val_accs = {name: {} for name in CANDIDATES}
    val_accs["E_baseline_vanilla"] = {}
    sweep_names = [f"sweep_conf_{c:.2f}" for c in THRESHOLD_SWEEP_CONFS]
    for sn in sweep_names:
        val_accs[sn] = {}

    print("\n" + "=" * 60 + "\nVALIDASYON (Adim 5/6) -- 3 sahne\n" + "=" * 60)
    for scene in VAL_SCENES:
        seq_dir = args.data_dir / scene
        gt_by_frame = load_mot_gt(seq_dir / "gt" / "gt.txt")
        fps = read_seq_fps(seq_dir)

        detector = YoloDetector(weights=checkpoint_str, conf_threshold=0.35, classes=[0])
        t0 = time.perf_counter()
        per_frame, frame_shape = _detect_and_embed_once(seq_dir, detector, extractor, args.limit_frames)
        det_elapsed = time.perf_counter() - t0
        det_fps = len(per_frame) / det_elapsed if det_elapsed > 0 else 0.0
        dummy_frame = np.zeros(frame_shape, dtype=np.uint8)
        print(f"\n[{scene}] {len(per_frame)} kare (fine-tuned, conf>=0.35 havuzu), det_fps={det_fps:.1f}")

        rows, accs = _run_candidates_on_scene(per_frame, gt_by_frame, fps, dummy_frame, CANDIDATES,
                                               scene, "validation", checkpoint_str, det_fps)
        val_rows.extend(rows)
        for name, acc in accs.items():
            val_accs[name][scene] = acc
        for row in rows:
            print(f"  [{row['candidate']}] P={row['precision']:.3f} R={row['recall']:.3f} "
                  f"MOTA={row['mota']:.3f} IDF1={row['idf1']:.3f} IDSW={row['idsw']}")

        if not args.skip_threshold_sweep:
            sweep_candidates = {
                f"sweep_conf_{c:.2f}": dict(conf=c, assoc_dist_ratio=SWEEP_ASSOC_DIST_RATIO,
                                             match_thresh=SWEEP_MATCH_THRESH, track_thresh=SWEEP_TRACK_THRESH)
                for c in THRESHOLD_SWEEP_CONFS
            }
            srows, saccs = _run_candidates_on_scene(per_frame, gt_by_frame, fps, dummy_frame, sweep_candidates,
                                                     scene, "validation_threshold_sweep", checkpoint_str, det_fps)
            val_rows.extend(srows)
            for name, acc in saccs.items():
                val_accs[name][scene] = acc

        # Candidate E: vanilla COCO, ayri bir GPU gecisi gerektirir (farkli agirlik dosyasi)
        baseline_detector = YoloDetector(weights=BASELINE_WEIGHTS, conf_threshold=0.35, classes=[0])
        per_frame_b, shape_b = _detect_and_embed_once(seq_dir, baseline_detector, extractor, args.limit_frames)
        dummy_b = np.zeros(shape_b, dtype=np.uint8)
        row_b, acc_b = run_one_config(per_frame_b, gt_by_frame, fps, dummy_frame=dummy_b,
                                       return_accumulator=True, **BASELINE_PARAMS)
        row_b.update(scene=scene, candidate="E_baseline_vanilla", split="validation",
                      checkpoint=BASELINE_WEIGHTS, detection_fps=None)
        val_rows.append(row_b)
        val_accs["E_baseline_vanilla"][scene] = acc_b
        print(f"  [E_baseline_vanilla] P={row_b['precision']:.3f} R={row_b['recall']:.3f} "
              f"MOTA={row_b['mota']:.3f} IDF1={row_b['idf1']:.3f} IDSW={row_b['idsw']}")

    # --- agregasyon (Adim 6): global TP/FP/FN + motmetrics generate_overall ---
    all_candidate_names = list(CANDIDATES) + ["E_baseline_vanilla"] + (sweep_names if not args.skip_threshold_sweep else [])
    val_aggregate_rows = []
    for name in all_candidate_names:
        rows_for_name = [r for r in val_rows if r["candidate"] == name]
        agg = aggregate_rows(rows_for_name, val_accs[name])
        agg.update(candidate=name, split="validation",
                    checkpoint=BASELINE_WEIGHTS if name == "E_baseline_vanilla" else checkpoint_str)
        val_aggregate_rows.append(agg)

    val_csv = args.output_dir / "phase4_3_validation_results.csv"
    val_json = args.output_dir / "phase4_3_validation_results.json"
    with val_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(val_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(val_rows)
    val_json.write_text(json.dumps(
        {"per_scene": val_rows, "aggregate": val_aggregate_rows}, indent=2), encoding="utf-8")
    print(f"\nYazildi: {val_csv}\nYazildi: {val_json}")

    # ================= ADIM 7/8: SECIM + KILIT =================
    print("\n" + "=" * 60 + "\nSECIM (Adim 7) -- SADECE validasyon sonuclarindan\n" + "=" * 60)
    selection = select_locked_config(val_aggregate_rows)
    locked = selection["locked"]
    print(f"En yuksek IDF1: {selection['best_idf1']['candidate']} (IDF1={selection['best_idf1']['idf1']:.3f})")
    print(f"En yuksek MOTA: {selection['best_mota']['candidate']} (MOTA={selection['best_mota']['mota']:.3f})")
    print(f"En dusuk IDSW: {selection['lowest_idsw']['candidate']} (IDSW={selection['lowest_idsw']['idsw']})")
    print(f"En yuksek Recall: {selection['best_recall']['candidate']} (R={selection['best_recall']['recall']:.3f})")
    print(f"Pareto-optimal kume: {[r['candidate'] for r in selection['pareto_set']]}")
    print(f"\n>>> KILITLENDI: {locked['candidate']} <<<")
    selection_path = args.output_dir / "phase4_3_selection.json"
    selection_path.write_text(json.dumps({
        "best_idf1": locked["candidate"],
        "best_mota": selection["best_mota"]["candidate"],
        "lowest_idsw": selection["lowest_idsw"]["candidate"],
        "best_recall": selection["best_recall"]["candidate"],
        "pareto_set": [r["candidate"] for r in selection["pareto_set"]],
        "locked_candidate": locked["candidate"],
        "locked_params": CANDIDATES.get(locked["candidate"]) or next(
            (dict(conf=c, assoc_dist_ratio=SWEEP_ASSOC_DIST_RATIO, match_thresh=SWEEP_MATCH_THRESH,
                  track_thresh=SWEEP_TRACK_THRESH) for c in THRESHOLD_SWEEP_CONFS
             if f"sweep_conf_{c:.2f}" == locked["candidate"]), None),
    }, indent=2), encoding="utf-8")
    print(f"Yazildi: {selection_path}")

    locked_params = json.loads(selection_path.read_text(encoding="utf-8"))["locked_params"]

    # ================= ADIM 9: NIHAI TEST (KILITLENDIKTEN SONRA, DEGISTIRILMEZ) =================
    print("\n" + "=" * 60 + "\nNIHAI TEST (Adim 9) -- SADECE degerlendirme, ayar YOK\n" + "=" * 60)
    test_configs = {
        "baseline_E": (BASELINE_WEIGHTS, BASELINE_PARAMS),
        "finetuned_default_D": (checkpoint_str, CANDIDATES["D_finetuned_default"]),
        "phase4_2_exploratory_A": (checkpoint_str, CANDIDATES["A_idf1_optimum_4_2"]),
        f"phase4_3_locked_{locked['candidate']}": (checkpoint_str, locked_params),
    }
    test_rows: list = []
    for scene in TEST_SCENES:
        seq_dir = args.data_dir / scene
        gt_by_frame = load_mot_gt(seq_dir / "gt" / "gt.txt")
        fps = read_seq_fps(seq_dir)

        detector = YoloDetector(weights=checkpoint_str, conf_threshold=0.35, classes=[0])
        t0 = time.perf_counter()
        per_frame, frame_shape = _detect_and_embed_once(seq_dir, detector, extractor, args.limit_frames)
        det_fps = len(per_frame) / (time.perf_counter() - t0) if len(per_frame) else 0.0
        dummy_frame = np.zeros(frame_shape, dtype=np.uint8)

        baseline_detector = YoloDetector(weights=BASELINE_WEIGHTS, conf_threshold=0.35, classes=[0])
        per_frame_b, shape_b = _detect_and_embed_once(seq_dir, baseline_detector, extractor, args.limit_frames)
        dummy_b = np.zeros(shape_b, dtype=np.uint8)

        for name, (weights, params) in test_configs.items():
            pf, df = (per_frame_b, dummy_b) if weights == BASELINE_WEIGHTS else (per_frame, dummy_frame)
            row = run_one_config(pf, gt_by_frame, fps, dummy_frame=df, **params)
            row.update(scene=scene, configuration=name, split="final_test", checkpoint=weights,
                       detection_fps=det_fps if weights != BASELINE_WEIGHTS else None)
            test_rows.append(row)
            print(f"  [{scene}][{name}] P={row['precision']:.3f} R={row['recall']:.3f} "
                  f"MOTA={row['mota']:.3f} IDF1={row['idf1']:.3f} IDSW={row['idsw']}")

    test_csv = args.output_dir / "phase4_3_final_test_results.csv"
    test_json = args.output_dir / "phase4_3_final_test_results.json"
    with test_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(test_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)
    test_json.write_text(json.dumps(test_rows, indent=2), encoding="utf-8")
    print(f"\nYazildi: {test_csv}\nYazildi: {test_json}")

    return val_rows, val_aggregate_rows, selection, test_rows


if __name__ == "__main__":
    main()
