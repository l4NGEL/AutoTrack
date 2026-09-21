"""Faz 4.2: Faz 4.1'in fine-tune edilmis detector'i icin UCTAN UCA (tespit +
takip) optimizasyon -- sadece recall degil, MOTA/IDF1/IDSW/precision/FP/FN
dengesi.

**Neden bu script var:** Faz 4.1, fine-tune edilmis modelin
`conf_threshold=0.35` (COCO-egitilmis baseline icin kalibre edilmis, hic
degistirilmemis) ile calistirildiginda yuksek-yogunluklu sahnede recall'i
buyuk olcude iyilestirdigini (%15.1->%68.9) ama ID-switch'i patlattigini
(15->300, FP 1416->8215) bulmustu -- ve bunun en olasi aciklamasi olarak
"conf_threshold artik bu modele gore kalibre degil" cikarimini yapip
TEST ETMEDEN birakmisti. Bu script tam olarak o testi yapiyor.

**Mevcut (Faz 4.1) parametreler -- bu script'in "baseline"i:**
  conf_threshold=0.35, NMS iou=0.7 (Ultralytics varsayilani, hic
  degistirilmedi), AdaptiveTracker: association_dist_ratio=0.08,
  match_thresh=0.3, track_thresh=0.5 (bkz. asagidaki onemli not).

**Onemli, tekrar eden bir bulgu (Faz 3.0'da ayni sekilde bulunmustu):**
`AdaptiveTracker.track_thresh` (varsayilan 0.5) `update()` icinde YOLO'nun
kendi conf_threshold'undan BAGIMSIZ, IKINCI bir sessiz esik uygular. Bu
sweep'in 0.5'in ALTINDAKI degerleri (0.20-0.45) gercekten test edebilmesi
icin `track_thresh=0.0` sabitlendi -- run_confidence_sweep.py'nin (Faz 3.0)
AYNI disiplini.

**Performans -- tek gecis, coklu-konfigurasyon filtreleme (run_confidence_
sweep.py'den, Faz 3.0):** tespit+embedding SADECE en dusuk esikte (0.20)
BIR KEZ cikarilir; hem confidence sweep'in daha yuksek esikleri HEM DE
association-parametre sweep'inin tamami bu TEK pass'in ciktisini filtreleyip
SADECE tracker'i (Kalman/Hungarian/ReID-skoru -- GPU'suz, ucuz) yeniden
calistirarak elde edilir. Boylece 2 sahne icin toplam sadece 2 GPU
tespit-gecisi yapilir, sweep'teki onlarca konfigurasyon SIFIR ek YOLO/OSNet
maliyetiyle degerlendirilir.

**SIZINTI/GENELLEME SINIRI -- acikca belirtilmeli:** bu script'in kullandigi
2 sahne (`uav0000137_00458_v`, `uav0000305_00000_v`) Faz 4.0'in DOKUNULMAMIS
degerlendirme sahneleridir -- egitimde HIC kullanilmadilar (bkz.
confirm_no_leakage). AMA bu sweep de AYNI 2 sahne uzerinde parametre SECIYOR
-- yani bu, kesin bir "kalan veri kumesi"nde hyperparametre secimi DEGIL,
kesif niteliginde bir hassasiyet analizi (exploratory sensitivity analysis).
VisDrone-MOT'un metadata'si zaten indirilmis 3 sahne DAHA var (`uav0000117_
02622_v`, `uav0000182_00000_v`, `uav0000268_05773_v`) ki bunlar gercek bir
train/val/test ayrimi icin kullanilabilir -- ama goruntuleri henuz
indirilmedi (yeni bir indirme, kullanicinin onayini gerektirir, bu script
BUNU YAPMAZ). Secilen konfigurasyon bu yuzden "bu 2 sahnede en iyi calisan"
olarak raporlanmali, "genellenmis en iyi konfigurasyon" olarak DEGIL.

Kullanim:
    python -m evaluation.optimize_finetuned_tracker --checkpoint checkpoints/yolo_visdrone_finetune/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from detection.yolo_detector import YoloDetector
from evaluation.mot_metrics import evaluate_sequence, load_mot_gt, read_seq_fps, summarize, xyxy_to_xywh
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker
from tracking.bytetrack_wrapper import iou as iou_xyxy

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "VisDrone"
REID_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "osnet_mot_finetuned" / "model.pth.tar"
BASELINE_WEIGHTS = "yolov8n.pt"
EVAL_SCENES = ["uav0000137_00458_v", "uav0000305_00000_v"]  # Faz 4.0'in DOKUNULMAMIS sahneleri
TRAIN_SCENES = ["uav0000086_00000_v", "uav0000339_00001_v"]  # Faz 4.1'in egitim sahneleri
VALIDATION_SCENES_NOT_DOWNLOADED = ["uav0000117_02622_v", "uav0000182_00000_v", "uav0000268_05773_v"]

CONF_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
LOWEST_CONF = min(CONF_THRESHOLDS)
IOU_MATCH_THRESH = 0.5  # occlusion_recovery_rate.py / mot_metrics.py ile AYNI konvansiyon

# Faz 4.1'in kullandigi, henuz optimize edilmemis degerler
CURRENT_ASSOC_DIST_RATIO = 0.08
CURRENT_MATCH_THRESH = 0.3
CURRENT_TRACK_THRESH_FOR_SWEEP = 0.0  # bkz. modul docstring'i -- sessiz ikinci esik devre disi


def confirm_no_leakage(train_scenes=TRAIN_SCENES, eval_scenes=EVAL_SCENES) -> bool:
    overlap = set(train_scenes) & set(eval_scenes)
    if overlap:
        raise SystemExit(f"SIZINTI: {overlap} hem egitimde hem degerlendirmede kullanilmis!")
    print(f"[sizinti kontrolu] Egitim {train_scenes} ile degerlendirme {eval_scenes} arasinda kesisim YOK.")
    return True


class _PrecomputedExtractor:
    """run_confidence_sweep.py (Faz 3.0) ile AYNI -- onceden cikarilmis
    embedding'leri, tracker'in bekledigi sirayla oldugu gibi geri verir."""

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def extract(self, frame, boxes_xyxy):
        return self._embeddings


def _detect_and_embed_once(seq_dir: Path, detector: YoloDetector, extractor, limit_frames=None):
    img_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    if limit_frames:
        img_paths = img_paths[:limit_frames]
    per_frame = []
    frame_shape = None
    for img_path in img_paths:
        frame = cv2.imread(str(img_path))
        if frame_shape is None:
            frame_shape = frame.shape
        detections = [d for d in detector.detect(frame) if d.cls_name == "person"]
        embeddings = extractor.extract(frame, [d.xyxy for d in detections]) if detections else []
        per_frame.append(list(zip(detections, embeddings)))
    return per_frame, frame_shape


def match_boxes_hungarian(gt_boxes_xyxy: list, det_boxes_xyxy: list, iou_thresh: float = IOU_MATCH_THRESH) -> tuple:
    """Tek karede GT<->ham-tespit Hungarian eslesmesi -- occlusion_recovery_
    rate._match_gt_to_hyp ile AYNI yontem (Hungarian + IoU>=0.5), kutu
    listeleri uzerinde calisan bagimsiz/saf bir kopya. Doner: (tp, fp, fn)."""
    if not gt_boxes_xyxy:
        return 0, len(det_boxes_xyxy), 0
    if not det_boxes_xyxy:
        return 0, 0, len(gt_boxes_xyxy)
    cost = np.empty((len(gt_boxes_xyxy), len(det_boxes_xyxy)), dtype=float)
    for i, g in enumerate(gt_boxes_xyxy):
        for j, d in enumerate(det_boxes_xyxy):
            cost[i, j] = 1.0 - iou_xyxy(g, d)
    row_idx, col_idx = linear_sum_assignment(cost)
    tp = sum(1 for r, c in zip(row_idx, col_idx) if cost[r, c] <= 1.0 - iou_thresh)
    return tp, len(det_boxes_xyxy) - tp, len(gt_boxes_xyxy) - tp


def prf1(tp: int, fp: int, fn: int) -> tuple:
    """precision/recall NaN yalnizca gercekten TANIMSIZ oldugunda (tp+fp==0
    veya tp+fn==0); precision=recall=0.0 GECERLI bir durumdur ve F1=0.0
    vermelidir -- '0.0 falsy oldugu icin nan don' hatasi burada BULUNUP
    duzeltildi (bkz. tests/test_optimize_finetuned_tracker.py)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if precision != precision or recall != recall:  # NaN kontrolu (NaN != NaN)
        f1 = float("nan")
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def run_one_config(per_frame_dets_embs, gt_by_frame, fps, conf, assoc_dist_ratio, match_thresh,
                    track_thresh=CURRENT_TRACK_THRESH_FOR_SWEEP, dummy_frame=None) -> dict:
    """Onceden cikarilmis (tespit,embedding) cifti uzerinde -- SIFIR ek GPU
    maliyetiyle -- TEK bir (conf, association) konfigurasyonunu calistirir."""
    tracker = AdaptiveTracker(
        fps=fps, track_thresh=track_thresh, match_thresh=match_thresh,
        association_dist_ratio=assoc_dist_ratio, weight_mode="adaptive",
        base_weights=(0.4, 0.3, 0.3), use_missed_trust=True,
        use_embedding_gallery=True, use_fps_scaling=True,
    )
    hyp_by_frame_xyxy = {}
    tp_total = fp_total = fn_total = 0
    t0 = time.perf_counter()
    for i, pairs in enumerate(per_frame_dets_embs, start=1):
        kept = [(d, e) for d, e in pairs if d.conf >= conf]
        dets = [d for d, _ in kept]
        embs = [e for _, e in kept]
        tracks = tracker.update(dets, dummy_frame, extractor=_PrecomputedExtractor(embs))
        # DIKKAT: butun track'ler dahil (missed>0 coasting/Kalman-tahmini dahil) --
        # run_confidence_sweep.py (Faz 3.0) ve occlusion_event_forensics.py (Faz 3.1/4.1)
        # ile AYNI konvansiyon. Once burada yanlislikla missed==0 filtresi vardi; bu,
        # Faz 4.1'in raporlanmis MOTA/IDF1/IDSW sayilarindan SESSIZCE farkli bir
        # degerlendirme protokolu uretiyordu (coasted kutular dusunce FP_trk ~yariya
        # dusuyor, MOTA farkli isaret aliyor) -- "degerlendirme metrik tanimini sessizce
        # degistirme" ihlali olurdu, bu yuzden proje-capinda kurulu konvansiyona geri
        # donduruldu ve sweep bastan calistirildi.
        hyp_by_frame_xyxy[i] = [(t.track_id, t.xyxy) for t in tracks]

        gt_rows = gt_by_frame.get(i, [])
        gt_boxes = [(x, y, x + w, y + h) for _, (x, y, w, h) in gt_rows]
        det_boxes = [d.xyxy for d in dets]
        tp, fp, fn = match_boxes_hungarian(gt_boxes, det_boxes)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    elapsed = time.perf_counter() - t0

    hyp_xywh = {f: [(tid, xyxy_to_xywh(box)) for tid, box in rows] for f, rows in hyp_by_frame_xyxy.items()}
    acc = evaluate_sequence(gt_by_frame, hyp_xywh)
    summary = summarize({"r": acc})
    r = summary.loc["r"]
    precision, recall, f1 = prf1(tp_total, fp_total, fn_total)
    n_frames = len(per_frame_dets_embs)

    return {
        "conf": conf, "assoc_dist_ratio": assoc_dist_ratio, "match_thresh": match_thresh,
        "track_thresh": track_thresh,
        "tp": tp_total, "fp_det": fp_total, "fn_det": fn_total,
        "precision": precision, "recall": recall, "f1": f1,
        "mota": float(r["mota"]), "idf1": float(r["idf1"]), "idsw": int(r["num_switches"]),
        "fp_trk": int(r["num_false_positives"]), "fn_trk": int(r["num_misses"]),
        "mt": int(r["mostly_tracked"]), "pt": int(r["partially_tracked"]), "ml": int(r["mostly_lost"]),
        "gt_obj": int(r["num_objects"]),
        "tracking_fps": n_frames / elapsed if elapsed > 0 else 0.0,
    }


def _assoc_grid(current_dist_ratio, current_match_thresh):
    """Kucuk, kontrollu bir sweep -- 'aramayi patlatma' talimatina uygun.
    Mesafe orani: mevcut / +25% / +50% (Faz 3.5, MOT17'de GENISLETMENIN
    (0.08->0.15, ~+90%) hicbir fayda saglamadan bilinen bir hatayi geri
    getirdigini bulmustu -- burada VisDrone'da DAHA KUCUK adimlarla
    yeniden, bagimsiz test ediliyor, eski sonuc varsayilmiyor). Match
    threshold: mevcut -0.1 / mevcut / +0.1."""
    dist_ratios = [round(current_dist_ratio, 4), round(current_dist_ratio * 1.25, 4), round(current_dist_ratio * 1.5, 4)]
    match_threshes = [round(max(0.0, current_match_thresh - 0.1), 4), round(current_match_thresh, 4), round(current_match_thresh + 0.1, 4)]
    return dist_ratios, match_threshes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Fine-tune edilmis YOLO checkpoint'i")
    parser.add_argument("--scenes", default=",".join(EVAL_SCENES))
    parser.add_argument("--conf-thresholds", default=",".join(str(c) for c in CONF_THRESHOLDS))
    parser.add_argument("--top-k-for-tracker-sweep", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "experiments")
    parser.add_argument("--limit-frames", type=int, default=None, help="Hizli duman testi icin")
    args = parser.parse_args()

    confirm_no_leakage()
    scenes = args.scenes.split(",")
    conf_thresholds = [float(c) for c in args.conf_thresholds.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reid_ckpt = str(REID_CHECKPOINT) if REID_CHECKPOINT.is_file() else None
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=reid_ckpt)

    all_rows = []
    cached_by_scene = {}  # scene -> (per_frame_dets_embs, frame_shape, fps, gt_by_frame) -- tekrar tespit YOK

    for scene in scenes:
        seq_dir = args.data_dir / scene
        gt_by_frame = load_mot_gt(seq_dir / "gt" / "gt.txt")
        fps = read_seq_fps(seq_dir)
        detector = YoloDetector(weights=args.checkpoint, conf_threshold=LOWEST_CONF, classes=[0])

        print(f"\n=== {scene}: tek GPU gecisi (conf>={LOWEST_CONF}) ===")
        per_frame_dets_embs, frame_shape = _detect_and_embed_once(seq_dir, detector, extractor, args.limit_frames)
        dummy_frame = np.zeros(frame_shape, dtype=np.uint8)
        cached_by_scene[scene] = (per_frame_dets_embs, frame_shape, dummy_frame, fps, gt_by_frame)
        total_dets = sum(len(p) for p in per_frame_dets_embs)
        print(f"  {len(per_frame_dets_embs)} kare, {total_dets} toplam ham tespit")

        print(f"\n--- ADIM 2: confidence sweep ({len(conf_thresholds)} deger) ---")
        for idx, conf in enumerate(conf_thresholds, start=1):
            row = run_one_config(per_frame_dets_embs, gt_by_frame, fps, conf,
                                  CURRENT_ASSOC_DIST_RATIO, CURRENT_MATCH_THRESH, dummy_frame=dummy_frame)
            row.update(scene=scene, stage="confidence_sweep", checkpoint=args.checkpoint)
            all_rows.append(row)
            print(f"  [{idx}/{len(conf_thresholds)}] conf={conf:.2f}  "
                  f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f}  "
                  f"MOTA={row['mota']:.3f} IDF1={row['idf1']:.3f} IDSW={row['idsw']}")

    # --- ADIM 3: her sahne icin en iyi (IDF1'e gore) top-K conf aday ---
    candidates_by_scene = {}
    for scene in scenes:
        scene_rows = [r for r in all_rows if r["scene"] == scene and r["stage"] == "confidence_sweep"]
        top = sorted(scene_rows, key=lambda r: r["idf1"], reverse=True)[: args.top_k_for_tracker_sweep]
        candidates_by_scene[scene] = [r["conf"] for r in top]
        print(f"\n[{scene}] IDF1'e gore top-{args.top_k_for_tracker_sweep} conf aday: {candidates_by_scene[scene]}")

    # --- ADIM 4/5: tracker (association) sweep -- SADECE aday conf'larda ---
    dist_ratios, match_threshes = _assoc_grid(CURRENT_ASSOC_DIST_RATIO, CURRENT_MATCH_THRESH)
    print(f"\n--- ADIM 4/5: tracker sweep -- assoc_dist_ratio={dist_ratios}, match_thresh={match_threshes} ---")
    for scene in scenes:
        per_frame_dets_embs, frame_shape, dummy_frame, fps, gt_by_frame = cached_by_scene[scene]
        combos = [(c, adr, mt) for c in candidates_by_scene[scene] for adr in dist_ratios for mt in match_threshes]
        for idx, (conf, adr, mt) in enumerate(combos, start=1):
            if adr == CURRENT_ASSOC_DIST_RATIO and mt == CURRENT_MATCH_THRESH:
                continue  # zaten confidence_sweep asamasinda olculdu, tekrar etme
            row = run_one_config(per_frame_dets_embs, gt_by_frame, fps, conf, adr, mt, dummy_frame=dummy_frame)
            row.update(scene=scene, stage="tracker_sweep", checkpoint=args.checkpoint)
            all_rows.append(row)
            print(f"  [{scene}][{idx}/{len(combos)}] conf={conf:.2f} dist_ratio={adr:.3f} match_thresh={mt:.2f}  "
                  f"MOTA={row['mota']:.3f} IDF1={row['idf1']:.3f} IDSW={row['idsw']}")

    # --- ADIM 7: baseline (vanilla COCO) karsilastirmasi -- ayni script/metrikler ---
    print("\n--- Baseline (vanilla yolov8n.pt, conf=0.35, mevcut association parametreleri) ---")
    for scene in scenes:
        seq_dir = args.data_dir / scene
        gt_by_frame = load_mot_gt(seq_dir / "gt" / "gt.txt")
        fps = read_seq_fps(seq_dir)
        baseline_detector = YoloDetector(weights=BASELINE_WEIGHTS, conf_threshold=0.35, classes=[0])
        per_frame, frame_shape = _detect_and_embed_once(seq_dir, baseline_detector, extractor, args.limit_frames)
        dummy_frame = np.zeros(frame_shape, dtype=np.uint8)
        row = run_one_config(per_frame, gt_by_frame, fps, 0.35, CURRENT_ASSOC_DIST_RATIO, CURRENT_MATCH_THRESH,
                              track_thresh=0.5, dummy_frame=dummy_frame)
        row.update(scene=scene, stage="baseline_vanilla", checkpoint=BASELINE_WEIGHTS)
        all_rows.append(row)
        print(f"  [{scene}] MOTA={row['mota']:.3f} IDF1={row['idf1']:.3f} IDSW={row['idsw']} "
              f"P={row['precision']:.3f} R={row['recall']:.3f}")

    # --- ciktilari yaz ---
    csv_path = args.output_dir / "phase4_2_optimization_results.csv"
    json_path = args.output_dir / "phase4_2_optimization_results.json"
    fieldnames = list(all_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    json_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"\nYazildi: {csv_path}\nYazildi: {json_path}")

    return all_rows


if __name__ == "__main__":
    main()
