"""Kullanicinin istegi: 'ID neden degisti' sorusuna tahmin degil, kod
seviyesinde kesin cevap. Demo videosunda (config_demo.yaml, guncel en iyi
pipeline) HER GT kimligin, zaman icinde hangi hyp track_id'lerle eslestigini
(Hungarian, IoU>=0.5, occlusion_recovery_rate.py ile ayni prensip) cikarir
-- boylece "hyp=A bir yerde bitti, hyp=B baska bir yerde basladi, ikisi
GERCEKTEN ayni GT kimlik mi?" sorusu varsayimla degil, dogrudan GT
zemininde cevaplanir.

Kullanim:
    python -m evaluation.trace_track_lifecycle
    python -m evaluation.trace_track_lifecycle --from-cache
"""
import argparse
import json
from pathlib import Path

import cv2

from detection.yolo_detector import YoloDetector
from evaluation.mot_metrics import read_seq_fps
from evaluation.occlusion_recovery_rate import DEFAULT_CHECKPOINT, _load_gt_all_visibility, _match_gt_to_hyp
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MOT17"
SEQUENCE = "MOT17-04-FRCNN"
CACHE_PATH = Path(__file__).resolve().parent / "_cache" / "track_lifecycle_MOT17-04-FRCNN.json"
# config_demo.yaml ile AYNI -- demo videosunda gercekten calisan konfigurasyon
BASELINE_CONF = 0.35
TRACK_THRESH = 0.5


def _collect(seq_dir: Path, fps: float, limit_frames=None) -> dict:
    """{frame: [(track_id, xyxy), ...]}"""
    detector = YoloDetector(weights="yolov8n.pt", conf_threshold=BASELINE_CONF, classes=[0])
    checkpoint_path = str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.is_file() else None
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=checkpoint_path)
    tracker = AdaptiveTracker(weight_mode="adaptive", base_weights=(0.4, 0.3, 0.3),
                               track_thresh=TRACK_THRESH, fps=fps)

    img_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    if limit_frames:
        img_paths = img_paths[:limit_frames]
    hyp_by_frame = {}
    for i, img_path in enumerate(img_paths, start=1):
        frame = cv2.imread(str(img_path))
        detections = detector.detect(frame)
        person_dets = [d for d in detections if d.cls_name == "person"]
        tracks = tracker.update(person_dets, frame, extractor=extractor)
        hyp_by_frame[i] = [(t.track_id, list(t.xyxy)) for t in tracks]
    return hyp_by_frame


def _build_gt_hyp_timeline(gt_by_frame_full: dict, hyp_by_frame: dict) -> dict:
    """{gt_id: [(frame, hyp_id_or_None), ...]} -- HER karede (gt gorunur olsun
    olmasin) Hungarian eslestirmesiyle. gt_id sadece gorunur oldugu karelerde
    listelenir (gorunmezken hyp aramak anlamsiz)."""
    n_frames = max(gt_by_frame_full.keys())
    timeline: dict = {}
    for frame in range(1, n_frames + 1):
        gt_rows = gt_by_frame_full.get(frame, [])
        hyp_rows = hyp_by_frame.get(frame, [])
        matches = _match_gt_to_hyp(gt_rows, [(tid, tuple(box)) for tid, box in hyp_rows])
        for gt_id, _box, vis in gt_rows:
            if vis >= 0.2:
                timeline.setdefault(gt_id, []).append((frame, matches.get(gt_id)))
    return timeline


def _segments(timeline_entries: list) -> list:
    """[(hyp_id_or_None, start_frame, end_frame), ...] -- ardisik ayni hyp_id'yi tek segmente birlestirir."""
    segs = []
    for frame, hyp_id in timeline_entries:
        if segs and segs[-1][0] == hyp_id:
            segs[-1] = (hyp_id, segs[-1][1], frame)
        else:
            segs.append((hyp_id, frame, frame))
    return segs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--min-switches", type=int, default=1,
                         help="En az bu kadar FARKLI (None olmayan) hyp_id degisimi olan GT kimlikleri goster")
    parser.add_argument("--limit-frames", type=int, default=None, help="Hizli duman testi icin ilk N kareyle sinirla")
    args = parser.parse_args()

    if args.from_cache:
        if not CACHE_PATH.is_file():
            raise SystemExit(f"Cache yok: {CACHE_PATH} -- once --from-cache OLMADAN calistirin")
        timeline = {int(k): v for k, v in json.loads(CACHE_PATH.read_text(encoding="utf-8")).items()}
    else:
        seq_dir = DATA_DIR / SEQUENCE
        gt_by_frame_full = _load_gt_all_visibility(seq_dir / "gt" / "gt.txt")
        seq_fps = read_seq_fps(seq_dir)
        print(f"=== {SEQUENCE} -- hyp uretiliyor (conf={BASELINE_CONF}, track_thresh={TRACK_THRESH}, config_demo.yaml ile ayni) ===")
        hyp_by_frame = _collect(seq_dir, seq_fps, args.limit_frames)
        print("=== GT<->hyp zaman cizelgesi cikariliyor (Hungarian, her kare) ===")
        timeline = _build_gt_hyp_timeline(gt_by_frame_full, hyp_by_frame)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps({str(k): v for k, v in timeline.items()}), encoding="utf-8")
        print(f"  Cache'e yazildi: {CACHE_PATH}")

    print(f"\n{'gt_id':>6s}  segmentler (hyp_id: baslangic-bitis kare)")
    interesting = []
    for gt_id, entries in sorted(timeline.items()):
        segs = _segments(entries)
        distinct_hyp = {h for h, _, _ in segs if h is not None}
        n_switches = max(0, len(distinct_hyp) - 1) if distinct_hyp else 0
        if n_switches >= args.min_switches:
            interesting.append((gt_id, segs, n_switches))

    for gt_id, segs, n_switches in interesting:
        seg_str = "  ".join(
            f"{'None' if h is None else h}:{s}-{e}" for h, s, e in segs
        )
        print(f"{gt_id:6d}  ({n_switches} hyp-degisimi) {seg_str}")

    print(f"\nToplam {len(interesting)} GT kimlik, en az {args.min_switches} hyp_id degisimiyle "
          f"(toplam {len(timeline)} gorunur GT kimlik icinden).")

    return timeline


if __name__ == "__main__":
    main()
