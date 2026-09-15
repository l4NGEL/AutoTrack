"""Faz 4.1: fine-tune edilmis YOLO checkpoint'ini, Faz 4.0'in DOKUNULMAMIS
2 degerlendirme sahnesinde (uav0000137_00458_v, uav0000305_00000_v),
baseline (vanilla COCO yolov8n.pt) ile dogrudan kiyaslar.

Kullanicinin istedigi tabloyu uretir: detection sayisi/recall, ID-switch,
"detector hic gormedi" orani (P1_NO_DETECTION'in VisDrone'a uyarlanmis
hali -- asagidaki NOT'a bakin), MOTA/IDF1, inference/FPS maliyeti.

**NOT -- P1_NO_DETECTION neden BIREBIR tasinamiyor:** partial_visibility_
forensics.py'nin P1-P4 taksonomisi, MOT17'nin SUREKLI (0.05-1.0) gorunurluk
degerine dayanir ve "partial visibility episode" (0.20<=vis<0.50, >=60 kare
surekli) kavramini KULLANIR. `visdrone_adapter.py`'nin docstring'i (madde 1)
bunun VisDrone'da yapisal olarak imkansiz oldugunu zaten belgeliyor:
VisDrone'un (bu mirror'daki) gorunurlugu IKILI (0/1), hicbir deger 0.2-0.5
araligina dusemez. Bu script o yuzden P1_NO_DETECTION'in ALTINDA yatan
gercek soruyu ("detector, gercekten orada olan bir insani hic gormuyor mu?")
DOGRUDAN, episode kavramindan bagimsiz olarak olcer: her GT kutusu icin,
ayni karede IoU>=0.5 ile eslesen HICBIR ham tespit yoksa bu bir "GT-miss"
sayilir. GT-miss orani = P1_NO_DETECTION'in olcmeye calistigi seyin
(detection recall) VisDrone'a tasinabilir, dogrudan analogu.

Kullanim:
    python -m evaluation.compare_finetune_vs_baseline --finetuned checkpoints/yolo_visdrone_finetune/weights/best.pt
"""
import argparse
import time
from pathlib import Path

import cv2

from detection.yolo_detector import YoloDetector
from evaluation.mot_metrics import evaluate_sequence, load_mot_gt, read_seq_fps, summarize, xyxy_to_xywh
from evaluation.occlusion_event_forensics import BASELINE_CONF, _collect_baseline_hyp
from tracking.bytetrack_wrapper import iou as iou_xyxy

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "VisDrone"
EVAL_SCENES = ["uav0000137_00458_v", "uav0000305_00000_v"]  # Faz 4.0'in degerlendirme sahneleri -- FINE-TUNE'DA HIC KULLANILMADI
TRAIN_SCENES = ["uav0000086_00000_v", "uav0000339_00001_v"]  # Faz 4.1'in egitim sahneleri


def confirm_no_leakage(train_scenes=TRAIN_SCENES, eval_scenes=EVAL_SCENES) -> bool:
    overlap = set(train_scenes) & set(eval_scenes)
    if overlap:
        raise SystemExit(f"SIZINTI: {overlap} hem egitimde hem degerlendirmede kullanilmis!")
    print(f"[sizinti kontrolu] Egitim sahneleri {train_scenes} ile degerlendirme "
          f"sahneleri {eval_scenes} arasinda kesisim YOK -- dogrulandi.")
    return True


def _count_gt_misses(gt_rows: list, det_boxes: list, iou_thresh: float = 0.5) -> tuple:
    """Saf eslesme mantigi -- I/O yok, birim testle dogrulanabilir. gt_rows:
    [(obj_id, (x,y,w,h)), ...] (load_mot_gt formati). det_boxes: [xyxy, ...].
    Doner: (bu karedeki toplam GT sayisi, IoU>=iou_thresh ile HICBIR tespite
    eslesmeyen GT sayisi)."""
    total, missed = 0, 0
    for _gt_id, (x, y, w, h) in gt_rows:
        total += 1
        gt_box = (x, y, x + w, y + h)
        best_iou = max((iou_xyxy(gt_box, db) for db in det_boxes), default=0.0)
        if best_iou < iou_thresh:
            missed += 1
    return total, missed


def _detection_recall(seq_dir: Path, weights: str, conf: float = BASELINE_CONF) -> dict:
    """Her GT kutusu icin, AYNI karede IoU>=0.5 ile eslesen HICBIR ham tespit
    yoksa 'GT-miss' sayar -- bkz. modul docstring'i (P1_NO_DETECTION'in
    VisDrone-uyumlu analogu). Takip KATMANI devre disi -- SADECE tek-kare
    detector cikisini olcer."""
    detector = YoloDetector(weights=weights, conf_threshold=conf, classes=[0])
    gt_by_frame = load_mot_gt(seq_dir / "gt" / "gt.txt")

    total_gt, missed_gt, total_dets = 0, 0, 0
    t0 = time.perf_counter()
    img_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    for img_path in img_paths:
        frame_number = int(img_path.stem)
        frame = cv2.imread(str(img_path))
        dets = [d for d in detector.detect(frame) if d.cls_name == "person"]
        total_dets += len(dets)

        gt_rows = gt_by_frame.get(frame_number, [])
        n, m = _count_gt_misses(gt_rows, [d.xyxy for d in dets])
        total_gt += n
        missed_gt += m
    elapsed = time.perf_counter() - t0

    return {
        "total_gt": total_gt,
        "gt_miss": missed_gt,
        "gt_miss_rate": missed_gt / total_gt if total_gt else float("nan"),
        "recall_at_iou50": 1 - (missed_gt / total_gt) if total_gt else float("nan"),
        "total_detections": total_dets,
        "fps": len(img_paths) / elapsed if elapsed > 0 else 0.0,
    }


def _tracking_metrics(seq_dir: Path, weights: str) -> dict:
    gt_by_frame = load_mot_gt(seq_dir / "gt" / "gt.txt")
    seq_fps = read_seq_fps(seq_dir)
    t0 = time.perf_counter()
    hyp_by_frame_xyxy = _collect_baseline_hyp(seq_dir, seq_fps, weights=weights)
    elapsed = time.perf_counter() - t0
    hyp_by_frame_xywh = {
        f: [(tid, xyxy_to_xywh(tuple(box))) for tid, box in rows] for f, rows in hyp_by_frame_xyxy.items()
    }
    acc = evaluate_sequence(gt_by_frame, hyp_by_frame_xywh)
    summary = summarize({"seq": acc})
    r = summary.loc["seq"]
    n_frames = len(hyp_by_frame_xyxy)
    return {
        "mota": float(r["mota"]), "idf1": float(r["idf1"]), "idsw": int(r["num_switches"]),
        "fp": int(r["num_false_positives"]), "fn": int(r["num_misses"]), "gt_obj": int(r["num_objects"]),
        "fps": n_frames / elapsed if elapsed > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", required=True, help="Fine-tune edilmis checkpoint yolu (orn. checkpoints/yolo_visdrone_finetune/weights/best.pt)")
    parser.add_argument("--baseline", default="yolov8n.pt")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    if not Path(args.finetuned).is_file():
        raise SystemExit(f"Fine-tune checkpoint bulunamadi: {args.finetuned}")

    confirm_no_leakage()

    all_rows = []
    for scene in EVAL_SCENES:
        seq_dir = args.data_dir / scene
        print(f"\n{'='*70}\n{scene}\n{'='*70}")
        for label, weights in [("baseline", args.baseline), ("fine-tuned", args.finetuned)]:
            print(f"\n--- {label} ({weights}) ---")
            print("  [1/2] tek-kare detection recall olculuyor...")
            det = _detection_recall(seq_dir, weights)
            print(f"    GT kutusu: {det['total_gt']}, GT-miss: {det['gt_miss']} "
                  f"({det['gt_miss_rate']:.1%}), recall@IoU0.5: {det['recall_at_iou50']:.1%}, "
                  f"toplam ham tespit: {det['total_detections']}, {det['fps']:.1f} FPS (sadece detector)")
            print("  [2/2] takip + MOTA/IDF1 olculuyor...")
            trk = _tracking_metrics(seq_dir, weights)
            print(f"    MOTA={trk['mota']:.3f} IDF1={trk['idf1']:.3f} IDSW={trk['idsw']} "
                  f"FP={trk['fp']} FN={trk['fn']} GT_obj={trk['gt_obj']} {trk['fps']:.1f} FPS (uctan-uca)")
            all_rows.append({"scene": scene, "config": label, **det, **{f"trk_{k}": v for k, v in trk.items()}})

    print(f"\n\n{'='*90}\nOZET TABLO (baseline -> fine-tuned)\n{'='*90}")
    header = (f"{'Sahne':22s} {'Config':12s} {'GT-miss%':>9s} {'Recall%':>8s} {'Ham tespit':>11s} "
              f"{'MOTA':>7s} {'IDF1':>7s} {'IDSW':>5s} {'FP':>6s} {'FN':>6s} {'Det.FPS':>8s} {'Uctan-uca FPS':>13s}")
    print(header)
    for row in all_rows:
        print(f"{row['scene']:22s} {row['config']:12s} {row['gt_miss_rate']*100:8.1f}% "
              f"{row['recall_at_iou50']*100:7.1f}% {row['total_detections']:11d} "
              f"{row['trk_mota']:7.3f} {row['trk_idf1']:7.3f} {row['trk_idsw']:5d} "
              f"{row['trk_fp']:6d} {row['trk_fn']:6d} {row['fps']:8.1f} {row['trk_fps']:13.1f}")

    return all_rows


if __name__ == "__main__":
    main()
