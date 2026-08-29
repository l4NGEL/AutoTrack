"""Occlusion Recovery Rate (ORR): occlusion sonrasi ayni GT kimligin ayni
tracker ID'sini koruma orani.

Tanim (kullanicinin belirttigi sekilde):
  occlusion event = bir GT kimligin en az `min_occlusion_frames` kare
  boyunca yok sayilmasi (gt.txt'de satiri yok VEYA gorunurluk<0.2), hem
  ONCESINDE hem SONRASINDA gercekten gorunur (vis>=0.2) olmasi sartiyla.

  Her event icin:
    before_hyp_id = occlusion'dan hemen once, bu GT kimlige IOU>=0.5 ile
                    eslesen tracker hipotez ID'si (birkac kare geriye bakar)
    after_hyp_id  = occlusion'dan hemen sonra, ayni sekilde eslesen ID
    recovered     = before_hyp_id == after_hyp_id (ikisi de bulunabildiyse)

  ORR = recovered / (before VE after bulunabilen) toplam event

`max_recovery_gap` (varsayilan 60 -- AdaptiveTracker'in reid_recovery_window'u
ile ayni) event'leri iki gruba ayirir: pencere ICINDE (sistemin tasarim geregi
kurtarmayi DENEMESI beklenir) ve pencere DISINDA (kurtarilmamasi TASARIM
GEREGI beklenir -- bkz. README "Faz 2.5" ID7/46 tartismasi). Ikisini ayri
raporlamak, "bug" ile "bilincli tasarim siniri"ni birbirine karistirmamak
icindir.

Kullanim:
    python -m evaluation.occlusion_recovery_rate --sequence MOT17-04-FRCNN
    python -m evaluation.occlusion_recovery_rate --sequence MOT17-04-FRCNN --checkpoint checkpoints/osnet_market1501/model/model.pth.tar-15
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from detection.yolo_detector import YoloDetector
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker
from tracking.bytetrack_wrapper import iou as iou_xyxy

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MOT17"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "osnet_mot_finetuned" / "model.pth.tar"
IOU_MATCH_THRESH = 0.5
SEARCH_WINDOW = 5  # before/after hyp_id ararken en fazla bu kadar kare geri/ileri bak


def _load_gt_all_visibility(gt_path) -> dict:
    """{frame: [(gt_id, xyxy, vis), ...]} -- SADECE class==1 (yaya), gorunurluk filtresi YOK
    (occlusion'i tespit etmek icin dusuk gorunurluklu satirlar da lazim)."""
    by_frame: dict = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            gt_id = int(float(parts[1]))
            x, y, w, h = (float(v) for v in parts[2:6])
            cls = int(float(parts[7])) if len(parts) > 7 else 1
            vis = float(parts[8]) if len(parts) > 8 else 1.0
            if cls != 1:
                continue
            by_frame.setdefault(frame, []).append((gt_id, (x, y, x + w, y + h), vis))
    return by_frame


def _match_gt_to_hyp(gt_rows, hyp_rows) -> dict:
    """Basit greedy IOU eslestirme (en yuksek IOU'dan baslar) -- {gt_id: hyp_id}."""
    pairs = []
    for gt_id, gt_box, _ in gt_rows:
        for hyp_id, hyp_box in hyp_rows:
            score = iou_xyxy(gt_box, hyp_box)
            if score >= IOU_MATCH_THRESH:
                pairs.append((score, gt_id, hyp_id))
    pairs.sort(reverse=True)
    matched_gt, matched_hyp, result = set(), set(), {}
    for score, gt_id, hyp_id in pairs:
        if gt_id in matched_gt or hyp_id in matched_hyp:
            continue
        result[gt_id] = hyp_id
        matched_gt.add(gt_id)
        matched_hyp.add(hyp_id)
    return result


def compute_orr(gt_by_frame_full: dict, hyp_by_frame_xyxy: dict, min_occlusion_frames: int, max_recovery_gap: int):
    n_frames = max(gt_by_frame_full.keys()) if gt_by_frame_full else 0

    # Her GT id icin: {frame: (visible_bool, gt_box)}
    identity_timeline: dict = {}
    for frame, rows in gt_by_frame_full.items():
        for gt_id, box, vis in rows:
            identity_timeline.setdefault(gt_id, {})[frame] = (vis >= 0.2, box)

    # Her kare icin gt_id -> hyp_id eslesmesi (sadece gorunur GT'ler icin anlamli)
    frame_matches: dict = {}
    for frame in range(1, n_frames + 1):
        gt_rows = gt_by_frame_full.get(frame, [])
        hyp_rows = hyp_by_frame_xyxy.get(frame, [])
        frame_matches[frame] = _match_gt_to_hyp(gt_rows, hyp_rows)

    def find_hyp_id_near(gt_id, center_frame, direction):
        for offset in range(0, SEARCH_WINDOW + 1):
            f = center_frame + direction * offset
            if f in frame_matches and gt_id in frame_matches[f]:
                return frame_matches[f][gt_id]
        return None

    events = []
    for gt_id, timeline in identity_timeline.items():
        frames_sorted = sorted(timeline.keys())
        if not frames_sorted:
            continue
        first_f, last_f = frames_sorted[0], frames_sorted[-1]

        i = first_f
        while i <= last_f:
            visible, _ = timeline.get(i, (False, None))
            if visible:
                i += 1
                continue
            # Gorunmezlik basladi -- ne kadar surecek bak
            gap_start = i
            while i <= last_f and not timeline.get(i, (False, None))[0]:
                i += 1
            gap_end = i - 1
            gap_len = gap_end - gap_start + 1

            before_visible = timeline.get(gap_start - 1, (False, None))[0]
            after_visible = i <= last_f and timeline.get(i, (False, None))[0]
            if gap_len >= min_occlusion_frames and before_visible and after_visible:
                before_hyp = find_hyp_id_near(gt_id, gap_start - 1, direction=-1)
                after_hyp = find_hyp_id_near(gt_id, gap_end + 1, direction=+1)
                events.append({
                    "gt_id": gt_id, "start": gap_start, "end": gap_end, "length": gap_len,
                    "before_hyp": before_hyp, "after_hyp": after_hyp,
                    "within_window": gap_len <= max_recovery_gap,
                })

    return events


def summarize_events(events: list, label: str):
    valid = [e for e in events if e["before_hyp"] is not None and e["after_hyp"] is not None]
    recovered = [e for e in valid if e["before_hyp"] == e["after_hyp"]]
    orr = len(recovered) / len(valid) if valid else float("nan")
    print(f"  [{label}] {len(events)} occlusion event bulundu, {len(valid)} gecerli "
          f"(oncesi+sonrasi eslesen), {len(recovered)} kurtarildi -> ORR={orr:.1%}" if valid
          else f"  [{label}] {len(events)} occlusion event, gecerli (once+sonra eslesen) yok")
    return {"label": label, "n_events": len(events), "n_valid": len(valid),
            "n_recovered": len(recovered), "orr": orr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="MOT17-04-FRCNN")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--min-occlusion-frames", type=int, default=5)
    parser.add_argument("--max-recovery-gap", type=int, default=60)
    args = parser.parse_args()

    seq_dir = DATA_DIR / args.sequence
    gt_by_frame_full = _load_gt_all_visibility(seq_dir / "gt" / "gt.txt")

    detector = YoloDetector(weights="yolov8n.pt", conf_threshold=0.35, classes=[0])
    checkpoint_path = str(args.checkpoint) if args.checkpoint.is_file() else None
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=checkpoint_path)
    tracker = AdaptiveTracker(weight_mode="adaptive", base_weights=(0.4, 0.3, 0.3))

    img_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    hyp_by_frame_xyxy = {}
    for i, img_path in enumerate(img_paths, start=1):
        frame = cv2.imread(str(img_path))
        detections = detector.detect(frame)
        person_dets = [d for d in detections if d.cls_name == "person"]
        tracks = tracker.update(person_dets, frame, extractor=extractor)
        hyp_by_frame_xyxy[i] = [(t.track_id, t.xyxy) for t in tracks]

    events = compute_orr(gt_by_frame_full, hyp_by_frame_xyxy, args.min_occlusion_frames, args.max_recovery_gap)

    print(f"\n=== {args.sequence} -- Occlusion Recovery Rate "
          f"(min_occlusion_frames={args.min_occlusion_frames}, max_recovery_gap={args.max_recovery_gap}) ===")
    within = [e for e in events if e["within_window"]]
    beyond = [e for e in events if not e["within_window"]]
    summarize_events(within, "pencere ICINDE (kurtarma beklenir)")
    summarize_events(beyond, "pencere DISINDA (tasarim geregi kurtarilmasi beklenmez)")


if __name__ == "__main__":
    main()
