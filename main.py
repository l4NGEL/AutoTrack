"""AutoTrack: gercek zamanli coklu hedef tespit, takip ve otonom onceliklendirme.

Kullanim:
    python main.py --config config.yaml
"""
import argparse

import cv2
import yaml

from detection.yolo_detector import YoloDetector
from tracking.bytetrack_wrapper import ByteTracker
from tracking.adaptive_tracker import AdaptiveTracker
from reid.feature_extractor import OSNetFeatureExtractor
from decision.priority_engine import PriorityEngine
from decision.zone_rules import load_zones
from viz.overlay_renderer import draw_zones, draw_tracks, draw_hud
from metrics.performance_logger import PerformanceLogger


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_tracker(tracker_cfg: dict):
    kind = tracker_cfg.get("type", "iou")
    if kind == "adaptive":
        a = tracker_cfg.get("adaptive", {})
        tracker = AdaptiveTracker(
            track_thresh=tracker_cfg["track_thresh"],
            track_buffer=tracker_cfg["track_buffer"],
            match_thresh=a.get("match_thresh", 0.3),
            weight_mode=a.get("weight_mode", "adaptive"),
            base_weights=tuple(a.get("base_weights", [0.4, 0.3, 0.3])),
            embedding_momentum=a.get("embedding_momentum", 0.9),
            reid_recovery_enabled=a.get("reid_recovery_enabled", True),
            reid_recovery_window=a.get("reid_recovery_window", 60),
            reid_recovery_thresh=a.get("reid_recovery_thresh", 0.75),
        )
    else:
        tracker = ByteTracker(
            track_thresh=tracker_cfg["track_thresh"],
            match_thresh=tracker_cfg["match_thresh"],
            track_buffer=tracker_cfg["track_buffer"],
        )
    return tracker, kind


def _open_capture(source):
    # MSMF bazi Windows webcam'lerinde frame yakalayamiyor; DirectShow'a dus.
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
    return cap


def run(cfg: dict):
    source = cfg["source"]
    cap = _open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Video kaynagi acilamadi: {source}")

    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(
            f"Video kaynagi acildi ama kare okunamadi: {source}. "
            "Kamera baska bir uygulama tarafindan kullaniliyor olabilir "
            "veya Windows kamera izinleri kapali olabilir."
        )
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) if not isinstance(source, int) else None

    detector = YoloDetector(
        weights=cfg["model_weights"],
        conf_threshold=cfg["conf_threshold"],
        classes=cfg.get("classes"),
    )
    tracker, tracker_kind = _build_tracker(cfg["tracker"])

    reid_extractor = None
    reid_cfg = cfg.get("reid", {})
    if tracker_kind == "adaptive" and reid_cfg.get("enabled", False):
        reid_extractor = OSNetFeatureExtractor(
            model_name=reid_cfg.get("model_name", "osnet_x1_0"),
            checkpoint_path=reid_cfg.get("checkpoint_path"),
            input_size=tuple(reid_cfg.get("input_size", (256, 128))),
        )

    zones = load_zones(cfg["zones"])
    engine = PriorityEngine(zones=zones, weights=cfg["priority_weights"])
    logger = PerformanceLogger(cfg["output"]["log_path"])

    writer = None
    if cfg["output"]["save_video"]:
        # mp4v disaridaki oynaticilarda (VLC, tarayici, ffprobe) sorunsuz
        # calisiyor, ama bu ortamdaki cv2.VideoCapture kendi yazdigi mp4v
        # dosyasini geri acamiyor (isOpened()=False) -- kareyi tekrar cv2
        # ile okumak gerekirse once ffmpeg ile H.264'e cevirin (bkz.
        # evaluation/make_demo_video.py'nin yaninda calistirilan ornek).
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width, height = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(cfg["output"]["output_path"], fourcc, 20.0, (width, height))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detector.detect(frame)
            if tracker_kind == "adaptive":
                tracks = tracker.update(detections, frame, extractor=reid_extractor)
            else:
                tracks = tracker.update(detections)

            priority_results = {}
            for track in tracks:
                priority_results[track.track_id] = engine.evaluate(track)

            fps = logger.tick()
            logger.log_frame(fps, len(detections), list(priority_results.values()), tracks)

            visible_count = sum(1 for t in tracks if getattr(t, "missed", 0) == 0)

            frame = draw_zones(frame, zones)
            frame = draw_tracks(frame, tracks, priority_results)
            frame = draw_hud(frame, fps, visible_count)  # draw_tracks ile tutarli olsun diye sadece bu karede goruleni say

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
