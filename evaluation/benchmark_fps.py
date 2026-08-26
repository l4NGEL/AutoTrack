"""Basit FPS/gecikme olcumu: gercek data/diag_*.png kareleri uzerinde
tekrarli calistirarak detection+tracking+ReID hattinin verimini olcer.

Not: Bu, MOT dogruluk metrikleri (HOTA/MOTA/IDF1) degildir -- sadece
throughput/gecikme olcumudur ve az sayida durgun kare uzerinde tekrarla
calisir. Dogruluk ablation'i icin evaluation/run_ablation.py (gercek bir
MOT dizisi gerektirir) kullanin.

Kullanim:
    python -m evaluation.benchmark_fps
"""
import statistics
import time
from pathlib import Path

import cv2

from detection.yolo_detector import YoloDetector
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker
from tracking.bytetrack_wrapper import ByteTracker

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPEATS = 30
WARMUP = 10


def _load_frames():
    paths = sorted(DATA_DIR.glob("diag_*.png"))
    frames = [cv2.imread(str(p)) for p in paths]
    return [f for f in frames if f is not None]


def _time_variant(name, run_once_fn, frames):
    for _ in range(WARMUP):
        run_once_fn(frames[0])

    latencies = []
    for _ in range(REPEATS):
        for frame in frames:
            t0 = time.perf_counter()
            run_once_fn(frame)
            latencies.append(time.perf_counter() - t0)

    avg_ms = statistics.mean(latencies) * 1000
    p95_ms = statistics.quantiles(latencies, n=20)[18] * 1000
    fps = 1000.0 / avg_ms
    print(f"{name:32s} avg={avg_ms:6.2f} ms  p95={p95_ms:6.2f} ms  FPS={fps:5.1f}  (n={len(latencies)})")
    return {"name": name, "avg_ms": round(avg_ms, 2), "p95_ms": round(p95_ms, 2), "fps": round(fps, 1)}


def main():
    frames = _load_frames()
    if not frames:
        print("data/diag_*.png bulunamadi, benchmark atlandi.")
        return []

    print(f"{len(frames)} gercek kare, her biri {REPEATS}x tekrarlanarak olculuyor "
          f"(toplam {len(frames) * REPEATS} cikarim/varyant)\n")

    detector = YoloDetector(weights="yolov8n.pt", conf_threshold=0.35)
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0")

    total_dets = sum(len(detector.detect(f)) for f in frames)
    if total_dets == 0:
        print(
            "UYARI: bu kareler (kamera diyagnostigi) hicbir nesne icermiyor -- "
            "'ReID acik' varyantinda OSNet hicbir kirpim uzerinde CALISTIRILMIYOR, "
            "yani bu olcum ReID'in gercek maliyetini YANSITMIYOR. Gercek kisi "
            "iceren kareler icin evaluation/run_ablation.py (MOT verisi) kullanin.\n"
        )

    results = []

    iou_tracker = ByteTracker()

    def run_iou(frame):
        dets = detector.detect(frame)
        iou_tracker.update(dets)

    results.append(_time_variant("IOU baseline (ByteTracker)", run_iou, frames))

    adaptive_no_reid = AdaptiveTracker()

    def run_adaptive_no_reid(frame):
        dets = detector.detect(frame)
        adaptive_no_reid.update(dets, frame, extractor=None)

    results.append(_time_variant("Adaptive, ReID kapali", run_adaptive_no_reid, frames))

    adaptive_with_reid = AdaptiveTracker()

    def run_adaptive_with_reid(frame):
        dets = detector.detect(frame)
        adaptive_with_reid.update(dets, frame, extractor=extractor)

    results.append(_time_variant("Adaptive, ReID acik (OSNet)", run_adaptive_with_reid, frames))

    return results


if __name__ == "__main__":
    main()
