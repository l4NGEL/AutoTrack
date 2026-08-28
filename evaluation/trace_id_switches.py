"""Demo videosunu tekrar isleyip her track_id'nin ilk/son karesini ve
konumunu kaydeder, sonra 'bir track kayboluyor, kisa sure sonra yakin bir
konumda yeni bir track dogruyor' orneklerini (olasi ID-switch) bulur.

Kullanim:
    python -m evaluation.trace_id_switches
    python -m evaluation.trace_id_switches --video data/demo_mot17_02_frcnn.mp4 --max-dist 300
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from detection.yolo_detector import YoloDetector
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker

DEFAULT_VIDEO = Path(__file__).resolve().parent.parent / "data" / "demo_mot17_04.mp4"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "osnet_mot_finetuned" / "model.pth.tar"


def center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-gap", type=int, default=45, help="Bu kadar kare icinde yakinda yeni track dogarsa aday say")
    parser.add_argument("--max-dist", type=float, default=120.0, help="Piksel cinsinden max merkez mesafesi")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = str(args.checkpoint) if args.checkpoint.is_file() else None
    if checkpoint is None:
        print(f"[trace] UYARI: checkpoint yok ({args.checkpoint}), ImageNet backbone kullanilacak")

    detector = YoloDetector(weights="yolov8n.pt", conf_threshold=0.35, classes=[0])
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=checkpoint)
    tracker = AdaptiveTracker()

    cap = cv2.VideoCapture(str(args.video))
    lifecycle = {}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        dets = detector.detect(frame)
        tracks = tracker.update(dets, frame, extractor=extractor)
        for t in tracks:
            if t.track_id not in lifecycle:
                lifecycle[t.track_id] = {"first": frame_idx, "first_box": t.xyxy, "last": frame_idx, "last_box": t.xyxy}
            else:
                lifecycle[t.track_id]["last"] = frame_idx
                lifecycle[t.track_id]["last_box"] = t.xyxy

    print(f"[trace] video: {args.video}")
    print(f"[trace] checkpoint: {checkpoint}")
    print(f"[trace] {frame_idx} kare islendi, {len(lifecycle)} benzersiz track_id")

    ended = sorted(lifecycle.items(), key=lambda kv: kv[1]["last"])
    started = sorted(lifecycle.items(), key=lambda kv: kv[1]["first"])

    print(f"\n=== Olasi ID-switch adaylari (gap<={args.max_gap}, dist<={args.max_dist:.0f}px) ===")
    candidates = []
    for tid_a, info_a in ended:
        cx_a, cy_a = center(info_a["last_box"])
        for tid_b, info_b in started:
            if tid_b == tid_a:
                continue
            gap = info_b["first"] - info_a["last"]
            if not (0 < gap <= args.max_gap):
                continue
            cx_b, cy_b = center(info_b["first_box"])
            dist = float(np.hypot(cx_b - cx_a, cy_b - cy_a))
            if dist <= args.max_dist:
                candidates.append((tid_a, tid_b, info_a["last"], info_b["first"], gap, dist))

    candidates.sort(key=lambda c: c[4])
    for tid_a, tid_b, last_a, first_b, gap, dist in candidates:
        box_a = lifecycle[tid_a]["last_box"]
        box_b = lifecycle[tid_b]["first_box"]
        print(f"  ID {tid_a} (son kare {last_a}, kutu {tuple(round(v) for v in box_a)}) -> "
              f"ID {tid_b} (ilk kare {first_b}, kutu {tuple(round(v) for v in box_b)})  "
              f"gap={gap} kare  merkez_mesafe={dist:.0f}px")

    if not candidates:
        print("  (bulunamadi -- esikleri gevsetmek gerekebilir)")

    out_path = args.out or (Path(__file__).resolve().parent.parent / "data" / "lifecycle_trace.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "video": str(args.video),
                "checkpoint": checkpoint,
                "lifecycle": {str(k): v for k, v in lifecycle.items()},
                "candidates": [
                    {"a": a, "b": b, "last_a": la, "first_b": fb, "gap": g, "dist_px": d}
                    for a, b, la, fb, g, d in candidates
                ],
            },
            f,
            indent=2,
        )
    print(f"\n[trace] tam veri kaydedildi: {out_path}")

    return lifecycle, candidates


if __name__ == "__main__":
    main()
