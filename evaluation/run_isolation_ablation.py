"""Faz 2.4 (missed-bazli mesafe daralmasi) ile Faz 2.5'i (embedding gallery)
BIRBIRINDEN AYRI olcen 4 kosullu ablation.

evaluation/run_ablation.py'nin E_Adaptive varyantini (weight_mode=adaptive,
base_weights=0.4/0.3/0.3, gercek ReID) sabit tutup sadece bu iki anahtari
degistirir:

  Baseline    : association_dist_ratio=0.15, use_missed_trust=False, use_embedding_gallery=False
                (Faz 2.4/2.5 oncesi davranis)
  2.4_only    : association_dist_ratio=0.08, use_missed_trust=True,  use_embedding_gallery=False
  2.5_only    : association_dist_ratio=0.15, use_missed_trust=False, use_embedding_gallery=True
  2.4+2.5     : association_dist_ratio=0.08, use_missed_trust=True,  use_embedding_gallery=True
                (mevcut varsayilan -- Faz 2.5 README'sindeki "Faz 2.4+2.5" satirlari)

Kullanim:
    python -m evaluation.run_isolation_ablation
"""
import time
from pathlib import Path

import cv2

from detection.yolo_detector import YoloDetector
from evaluation.mot_metrics import evaluate_sequence, format_summary, load_mot_gt, summarize, xyxy_to_xywh
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MOT17"
CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "osnet_mot_finetuned" / "model.pth.tar"
SEQUENCES = ["MOT17-02-FRCNN", "MOT17-04-FRCNN"]

CONDITIONS = {
    "Baseline (2.4/2.5 oncesi)": dict(association_dist_ratio=0.15, use_missed_trust=False, use_embedding_gallery=False),
    "2.4_only (mesafe)": dict(association_dist_ratio=0.08, use_missed_trust=True, use_embedding_gallery=False),
    "2.5_only (galeri)": dict(association_dist_ratio=0.15, use_missed_trust=False, use_embedding_gallery=True),
    "2.4+2.5 (guncel)": dict(association_dist_ratio=0.08, use_missed_trust=True, use_embedding_gallery=True),
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
    elapsed = time.perf_counter() - t0
    fps = len(img_paths) / elapsed if elapsed > 0 else 0.0
    return hyp_by_frame, fps


def main():
    detector = YoloDetector(weights="yolov8n.pt", conf_threshold=0.35, classes=[0])
    checkpoint_path = str(CHECKPOINT) if CHECKPOINT.is_file() else None
    if checkpoint_path is None:
        print(f"[run_isolation_ablation] UYARI: checkpoint yok ({CHECKPOINT})")
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=checkpoint_path)

    all_rows = []
    for seq in SEQUENCES:
        seq_dir = DATA_DIR / seq
        gt_by_frame = load_mot_gt(seq_dir / "gt" / "gt.txt")

        print(f"\n=== {seq} ===")
        accumulators, fps_by_cond = {}, {}
        for name, toggles in CONDITIONS.items():
            hyp_by_frame, fps = _run_tracker_on_sequence(seq_dir, detector, extractor, toggles)
            accumulators[name] = evaluate_sequence(gt_by_frame, hyp_by_frame)
            fps_by_cond[name] = fps
            print(f"  {name:28s} tamamlandi ({fps:.1f} FPS)")

        summary = summarize(accumulators)
        print(format_summary(summary))

        for name in CONDITIONS:
            row = summary.loc[name]
            all_rows.append({
                "sequence": seq, "condition": name,
                "mota": float(row["mota"]), "idf1": float(row["idf1"]),
                "id_switches": int(row["num_switches"]), "fps": round(fps_by_cond[name], 1),
            })

    print("\n\n=== TUM SONUCLAR (ozet) ===")
    print(f"{'Sekans':16s} {'Kosul':28s} {'MOTA':>7s} {'IDF1':>7s} {'ID Sw.':>7s} {'FPS':>6s}")
    for r in all_rows:
        print(f"{r['sequence']:16s} {r['condition']:28s} {r['mota']:7.3f} {r['idf1']:7.3f} "
              f"{r['id_switches']:7d} {r['fps']:6.1f}")

    return all_rows


if __name__ == "__main__":
    main()
