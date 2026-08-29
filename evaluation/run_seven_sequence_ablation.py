"""MOT17'nin FRCNN varyantindaki 7 sekansin TUMUNDE, 3 kosullu (Baseline /
sadece Faz 2.4 / Faz 2.4+2.5) ablation.

evaluation/run_isolation_ablation.py'nin 2-sekans/4-kosul denemesini
genisletir -- kullanicinin onerisi uzerine, tek-sekanslik sonuclarin
sekansa bagimli olabilecegini gormek icin.

ONEMLI -- temiz/kirli sekans ayrimi:
MOT17-09/11, Re-ID checkpoint'inin fine-tune EGITIMINDE kullanildi (bkz.
evaluation/finetune_reid_mot.py, evaluation/extract_mot_reid_crops.py).
Bu iki sekansta olculen Re-ID performansi domain-gap acisindan iyimser
olabilir (model o goruntuleri zaten gordu) -- sonuclar diger 5 "temiz"
(hic egitimde kullanilmamis) sekanstan AYRI raporlanir, aggregate sadece
temiz 5 sekans uzerinden hesaplanir.

Kullanim:
    python -m evaluation.run_seven_sequence_ablation
"""
import time
from pathlib import Path

import cv2

from detection.yolo_detector import YoloDetector
from evaluation.mot_metrics import evaluate_sequence, load_mot_gt, summarize, xyxy_to_xywh
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MOT17"
CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "osnet_mot_finetuned" / "model.pth.tar"

CLEAN_SEQUENCES = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-10-FRCNN", "MOT17-13-FRCNN"]
TAINTED_SEQUENCES = ["MOT17-09-FRCNN", "MOT17-11-FRCNN"]  # Re-ID fine-tune'de goruldu
ALL_SEQUENCES = CLEAN_SEQUENCES + TAINTED_SEQUENCES

CONDITIONS = {
    "A_Baseline": dict(association_dist_ratio=0.15, use_missed_trust=False, use_embedding_gallery=False),
    "B_Faz2.4_only": dict(association_dist_ratio=0.08, use_missed_trust=True, use_embedding_gallery=False),
    "C_Faz2.4+2.5": dict(association_dist_ratio=0.08, use_missed_trust=True, use_embedding_gallery=True),
}


def _run_tracker_on_sequence(seq_dir: Path, detector, extractor, toggles) -> dict:
    tracker = AdaptiveTracker(weight_mode="adaptive", base_weights=(0.4, 0.3, 0.3), **toggles)
    img_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    hyp_by_frame = {}
    t0 = time.perf_counter()
    for i, img_path in enumerate(img_paths, start=1):
        frame = cv2.imread(str(img_path))
        detections = detector.detect(frame)
        person_dets = [d for d in detections if d.cls_name == "person"]
        tracks = tracker.update(person_dets, frame, extractor=extractor)
        hyp_by_frame[i] = [(t.track_id, xyxy_to_xywh(t.xyxy)) for t in tracks]
    fps = len(img_paths) / max(time.perf_counter() - t0, 1e-6)
    return hyp_by_frame, fps


def main():
    detector = YoloDetector(weights="yolov8n.pt", conf_threshold=0.35, classes=[0])
    checkpoint_path = str(CHECKPOINT) if CHECKPOINT.is_file() else None
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=checkpoint_path)

    rows = []
    for seq in ALL_SEQUENCES:
        seq_dir = DATA_DIR / seq
        if not (seq_dir / "gt" / "gt.txt").is_file():
            print(f"[7seq] UYARI: {seq} indirilmemis, atlaniyor")
            continue
        gt_by_frame = load_mot_gt(seq_dir / "gt" / "gt.txt")
        tag = "temiz" if seq in CLEAN_SEQUENCES else "KIRLI(fine-tune'de goruldu)"
        print(f"\n=== {seq} [{tag}] ===")

        accumulators = {}
        for name, toggles in CONDITIONS.items():
            hyp_by_frame, fps = _run_tracker_on_sequence(seq_dir, detector, extractor, toggles)
            accumulators[name] = evaluate_sequence(gt_by_frame, hyp_by_frame)
            print(f"  {name:16s} tamamlandi ({fps:.1f} FPS)")

        summary = summarize(accumulators)
        for name in CONDITIONS:
            r = summary.loc[name]
            rows.append({
                "sequence": seq, "clean": seq in CLEAN_SEQUENCES, "condition": name,
                "mota": float(r["mota"]), "idf1": float(r["idf1"]),
                "id_switches": int(r["num_switches"]), "fp": int(r["num_false_positives"]),
                "fn": int(r["num_misses"]), "gt_obj": int(r["num_objects"]),
            })

    print(f"\n\n{'Sekans':16s} {'':22s} {'Kosul':16s} {'MOTA':>7s} {'IDF1':>7s} {'IDSw':>5s} {'FP':>6s} {'FN':>7s}")
    for r in rows:
        tag = "temiz" if r["clean"] else "KIRLI"
        print(f"{r['sequence']:16s} [{tag:5s}]       {r['condition']:16s} {r['mota']:7.3f} {r['idf1']:7.3f} "
              f"{r['id_switches']:5d} {r['fp']:6d} {r['fn']:7d}")

    # Aggregate -- SADECE temiz sekanslar (obj-sayisi agirlikli ortalama, ID-switch/FP/FN toplam)
    print("\n=== AGGREGATE (sadece 5 temiz sekans, GT-obj agirlikli MOTA/IDF1) ===")
    for name in CONDITIONS:
        clean_rows = [r for r in rows if r["condition"] == name and r["clean"]]
        total_obj = sum(r["gt_obj"] for r in clean_rows)
        w_mota = sum(r["mota"] * r["gt_obj"] for r in clean_rows) / total_obj if total_obj else 0.0
        w_idf1 = sum(r["idf1"] * r["gt_obj"] for r in clean_rows) / total_obj if total_obj else 0.0
        total_idsw = sum(r["id_switches"] for r in clean_rows)
        total_fp = sum(r["fp"] for r in clean_rows)
        total_fn = sum(r["fn"] for r in clean_rows)
        print(f"  {name:16s} MOTA={w_mota:.3f}  IDF1={w_idf1:.3f}  ID-Sw(toplam)={total_idsw}  "
              f"FP(toplam)={total_fp}  FN(toplam)={total_fn}")

    return rows


if __name__ == "__main__":
    main()
