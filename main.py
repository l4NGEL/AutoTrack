"""AutoTrack: gercek zamanli coklu hedef tespit, takip ve otonom onceliklendirme.

Kullanim:
    python main.py --config config.yaml
"""
import argparse

import cv2
import yaml

from detection.yolo_detector import YoloDetector
from tracking.bytetrack_wrapper import ByteTracker
from decision.priority_engine import PriorityEngine
from decision.zone_rules import load_zones
from viz.overlay_renderer import draw_zones, draw_tracks, draw_hud
from metrics.performance_logger import PerformanceLogger


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cfg: dict):
    source = cfg["source"]
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Video kaynagi acilamadi: {source}")

    detector = YoloDetector(
        weights=cfg["model_weights"],
        conf_threshold=cfg["conf_threshold"],
        classes=cfg.get("classes"),
    )
    tracker = ByteTracker(**cfg["tracker"])
    zones = load_zones(cfg["zones"])
    engine = PriorityEngine(zones=zones, weights=cfg["priority_weights"])
    logger = PerformanceLogger(cfg["output"]["log_path"])

    writer = None
    if cfg["output"]["save_video"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width, height = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(cfg["output"]["output_path"], fourcc, 20.0, (width, height))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections)

            priority_results = {}
            for track in tracks:
                priority_results[track.track_id] = engine.evaluate(track)

            fps = logger.tick()
            logger.log_frame(fps, len(detections), list(priority_results.values()))

            frame = draw_zones(frame, zones)
            frame = draw_tracks(frame, tracks, priority_results)
            frame = draw_hud(frame, fps, len(tracks))

            if writer:
                writer.write(frame)

            cv2.imshow("AutoTrack", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"Ortalama FPS: {logger.avg_fps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoTrack calistir")
    parser.add_argument("--config", default="config.yaml", help="Konfigurasyon dosyasi yolu")
    args = parser.parse_args()

    run(load_config(args.config))
