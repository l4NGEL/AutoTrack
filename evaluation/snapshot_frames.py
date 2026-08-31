"""Belirli kare numaralarinda, TAM overlay (main.py'nin cizdigi ayni ID/P/bucket
etiketleriyle) PNG kaydeder -- kullanicinin ekran goruntusuyle DOGRUDAN
gorsel karsilastirma icin (tahmin degil).

Kullanim:
    python -m evaluation.snapshot_frames --frames 95,96,97,98,99,100,101,102,495,496,497,498,499,500,501,502
"""
import argparse
from pathlib import Path

import cv2

from decision.priority_engine import PriorityEngine
from detection.yolo_detector import YoloDetector
from evaluation.mot_metrics import read_seq_fps
from evaluation.occlusion_recovery_rate import DEFAULT_CHECKPOINT
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker
from viz.overlay_renderer import draw_tracks

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MOT17"
SEQUENCE = "MOT17-04-FRCNN"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"
BASELINE_CONF = 0.35
TRACK_THRESH = 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="Virgulle ayrilmis kare numaralari (1-index, config_demo.yaml sekansiyla ayni)")
    args = parser.parse_args()
    target_frames = {int(x) for x in args.frames.split(",")}

    seq_dir = DATA_DIR / SEQUENCE
    seq_fps = read_seq_fps(seq_dir)

    detector = YoloDetector(weights="yolov8n.pt", conf_threshold=BASELINE_CONF, classes=[0])
    checkpoint_path = str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.is_file() else None
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=checkpoint_path)
    tracker = AdaptiveTracker(weight_mode="adaptive", base_weights=(0.4, 0.3, 0.3),
                               track_thresh=TRACK_THRESH, fps=seq_fps)
    engine = PriorityEngine(zones=[], weights={"speed": 0.4, "zone": 0.4, "dwell_time": 0.2}, fps=seq_fps)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    max_frame = max(target_frames)

    for i, img_path in enumerate(img_paths[:max_frame], start=1):
        frame = cv2.imread(str(img_path))
        detections = detector.detect(frame)
        person_dets = [d for d in detections if d.cls_name == "person"]
        tracks = tracker.update(person_dets, frame, extractor=extractor)

        if i in target_frames:
            priority_results = {t.track_id: engine.evaluate(t) for t in tracks}
            annotated = draw_tracks(frame.copy(), tracks, priority_results)
            out_path = OUT_DIR / f"frame_{i:06d}.png"
            cv2.imwrite(str(out_path), annotated)
            print(f"kare {i}: {len(tracks)} track -> {out_path}")


if __name__ == "__main__":
    main()
