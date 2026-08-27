"""MOT17 sekanslari uzerinde gercek ablation calismasi.

Ayni AdaptiveTracker sinifini farkli agirlik stratejileriyle calistirarak
A. IOU-only, B. IOU+Motion, C. ReID-only, D. Motion+ReID, E. Adaptive
varyantlarini karsilastirir. Hepsi ayni Hungarian-atama koduna girdigi icin
karsilastirma adil: fark sadece agirliklandirma stratejisinde.

Zemin dogrusu: MOT17 gt.txt (yaya sinifi, visibility>=0.2).
Tespit: kendi YOLOv8 dedektorumuz (MOT'un saglanmis det.txt'i degil --
AutoTrack uctan uca kendi dedektorunu kullanan bir sistem, o yuzden
degerlendirme de onu kullaniyor).

Kullanim:
    python -m evaluation.run_ablation
"""
import time
from pathlib import Path

import cv2

from detection.yolo_detector import YoloDetector
from evaluation.mot_metrics import evaluate_sequence, format_summary, load_mot_gt, summarize, xyxy_to_xywh
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker
from tracking.bytetrack_wrapper import ByteTracker

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MOT17"
CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "osnet_market1501" / "model" / "model.pth.tar-15"

SEQUENCES = ["MOT17-02-FRCNN", "MOT17-04-FRCNN"]

VARIANTS = {
    "A_IOU_only": dict(weight_mode="fixed", base_weights=(1.0, 0.0, 0.0), use_reid=False),
    "B_IOU_Motion": dict(weight_mode="fixed", base_weights=(0.5, 0.5, 0.0), use_reid=False),
    "C_ReID_only": dict(weight_mode="fixed", base_weights=(0.15, 0.0, 0.85), use_reid=True),
    "D_Motion_ReID": dict(weight_mode="fixed", base_weights=(0.0, 0.5, 0.5), use_reid=True),
    "E_Adaptive": dict(weight_mode="adaptive", base_weights=(0.4, 0.3, 0.3), use_reid=True),
}


def _run_tracker_on_sequence(seq_dir: Path, detector, extractor, variant_cfg) -> dict:
    tracker = AdaptiveTracker(
        weight_mode=variant_cfg["weight_mode"],
        base_weights=variant_cfg["base_weights"],
    )
    active_extractor = extractor if variant_cfg["use_reid"] else None

    img_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    hyp_by_frame = {}
    t0 = time.perf_counter()
    for i, img_path in enumerate(img_paths, start=1):
        frame = cv2.imread(str(img_path))
        detections = detector.detect(frame)
        person_dets = [d for d in detections if d.cls_name == "person"]
        tracks = tracker.update(person_dets, frame, extractor=active_extractor)
        hyp_by_frame[i] = [(t.track_id, xyxy_to_xywh(t.xyxy)) for t in tracks]
    elapsed = time.perf_counter() - t0
    fps = len(img_paths) / elapsed if elapsed > 0 else 0.0
    return hyp_by_frame, fps


def main():
    detector = YoloDetector(weights="yolov8n.pt", conf_threshold=0.35, classes=[0])  # 0 = person (COCO)

    checkpoint_path = str(CHECKPOINT) if CHECKPOINT.is_file() else None
    if checkpoint_path is None:
        print(f"[run_ablation] UYARI: egitilmis checkpoint bulunamadi ({CHECKPOINT}), "
              f"ImageNet backbone kullanilacak -- ReID varyantlari zayif kalir.")
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=checkpoint_path)

    all_rows = []
    for seq in SEQUENCES:
        seq_dir = DATA_DIR / seq
        gt_by_frame = load_mot_gt(seq_dir / "gt" / "gt.txt")

        print(f"\n=== {seq} ({len(list((seq_dir / 'img1').glob('*.jpg')))} kare) ===")
        accumulators = {}
        fps_by_variant = {}
        for name, cfg in VARIANTS.items():
            hyp_by_frame, fps = _run_tracker_on_sequence(seq_dir, detector, extractor, cfg)
            accumulators[name] = evaluate_sequence(gt_by_frame, hyp_by_frame)
            fps_by_variant[name] = fps
            print(f"  {name:16s} tamamlandi ({fps:.1f} FPS)")

        summary = summarize(accumulators)
        print(format_summary(summary))

        for name in VARIANTS:
            row = summary.loc[name]
            all_rows.append(
                {
                    "sequence": seq,
                    "variant": name,
                    "mota": float(row["mota"]),
                    "idf1": float(row["idf1"]),
                    "id_switches": int(row["num_switches"]),
                    "fps": round(fps_by_variant[name], 1),
                }
            )

    print("\n\n=== TUM SONUCLAR (ozet) ===")
    header = f"{'Sekans':16s} {'Varyant':16s} {'MOTA':>7s} {'IDF1':>7s} {'ID Sw.':>7s} {'FPS':>6s}"
    print(header)
    for r in all_rows:
        print(f"{r['sequence']:16s} {r['variant']:16s} {r['mota']:7.3f} {r['idf1']:7.3f} "
              f"{r['id_switches']:7d} {r['fps']:6.1f}")

    return all_rows


if __name__ == "__main__":
    main()
