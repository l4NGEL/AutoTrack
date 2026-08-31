"""Faz 3.3: Partial Visibility Persistence Forensics -- Faz 3.2'de bulunan
YENI hata modunun (gt_id=3, MOT17-04: GT gorunurlugu hicbir zaman 0.2 altina
dusmeden 184 kare boyunca hyp kaybi) ne kadar YAYGIN oldugunu olcer.

**BILINCLI TASARIM KARARI (kullanicinin istegi):** bu, ORR'un ("agir
occlusion", vis<0.2) tanimini SESSIZCE genisletmiyor -- AYRI bir event
tipi, AYRI bir script, AYRI metrikler:

  PARTIAL_VISIBILITY event = bir GT kimligin, ARDISIK en az
  `MIN_PARTIAL_DURATION` (60 -- reid_recovery_window ile ayni) kare
  boyunca, HER karede 0.20<=vis<0.50 kalmasi (0.20 altina dusen VEYA
  0.50'ye cikan bir kare varsa pencere biter -- bu "agir occlusion" ya da
  "normal gorunur" rejimine geciyor demektir).

Her episode, hyp_id gecmisine gore 4 sonuca ayrilir:
  RETAINED       -- episode boyunca HER karede ayni hyp_id (hic bosluk yok)
  RECOVERED      -- bosluk(lar) var ama hep AYNI hyp_id'ye donuluyor
  FRAGMENTED     -- episode icinde FARKLI bir hyp_id ortaya cikiyor (gercek,
                    "agir occlusion olmadan" ID-switch -- Faz 3.2'nin bulgusu)
  PERMANENT_LOSS -- episode boyunca HICBIR karede eslesme yok

PVTR (Partial Visibility Track Retention Rate) = (RETAINED+RECOVERED) / toplam
Fragmentation Rate = FRAGMENTED / toplam

FRAGMENTED/PERMANENT_LOSS episode'ler icin, kaybin BASLADIGI karede (son
eslesen kareden hemen sonraki) neden basarisiz oldugu P1-P4'e ayrilir
(occlusion_event_forensics.py'nin A/B/C/E taksonomisiyle AYNI ruhta, ama
P5/P6 kullanicinin onerisinde episode-sonucu seviyesiyle ortustugu icin
BILINCLI OLARAK BIRLESTIRILDI -- asagida aciklandi):
  P1 NO_DETECTION       -- o karede (conf>=0.15) hic tespit yok
  P2 LOW_IOU            -- tespit var ama en iyisi IoU<0.5
  P3 TRACK_THRESH_GATE  -- IoU>=0.5 AMA tespitin guveni tracker'in kendi
                          `track_thresh` esiginin (0.5) altinda -- Hungarian'a
                          hic girmeden reddediliyor
  P4 ASSIGNMENT_CONFLICT -- IoU>=0.5 VE guven>=track_thresh, ama yine de
                          Hungarian'da bu GT'ye eslenmemis (baska GT kapmis)

Kullanim:
    python -m evaluation.partial_visibility_forensics --sequence MOT17-04-FRCNN
    python -m evaluation.partial_visibility_forensics --sequence MOT17-04-FRCNN --from-cache
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from detection.yolo_detector import YoloDetector
from evaluation.mot_metrics import read_seq_fps
from evaluation.occlusion_event_forensics import _collect_boundary_detections
from evaluation.occlusion_recovery_rate import DEFAULT_CHECKPOINT, IOU_MATCH_THRESH, _load_gt_all_visibility, _match_gt_to_hyp
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker
from tracking.bytetrack_wrapper import iou as iou_xyxy

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MOT17"
CACHE_DIR = Path(__file__).resolve().parent / "_cache"
DEFAULT_SEQUENCE = "MOT17-04-FRCNN"
BASELINE_CONF = 0.35
TRACK_THRESH = 0.5  # AdaptiveTracker'in KENDI gating esigi -- P3 icin
LOW_CONF = 0.15
MIN_PARTIAL_DURATION = 60  # reid_recovery_window ile ayni
VIS_LOW, VIS_HIGH = 0.20, 0.50


def find_partial_visibility_episodes(gt_by_frame_full: dict, min_duration: int = MIN_PARTIAL_DURATION) -> list:
    """Her GT kimlik icin, ARDISIK 0.20<=vis<0.50 kare pencerelerini bulur
    (min_duration+ uzunlugunda olanlar). GT satiri YOKSA (hic annotate
    edilmemis) o kare bandin DISINDA sayilir (vis=0 kabul, "agir occlusion"
    rejimine dusuyor -- occlusion_recovery_rate.py ile AYNI konvansiyon)."""
    identity_timeline: dict = {}
    for frame, rows in gt_by_frame_full.items():
        for gt_id, box, vis in rows:
            identity_timeline.setdefault(gt_id, {})[frame] = (vis, box)

    episodes = []
    for gt_id, timeline in identity_timeline.items():
        frames_sorted = sorted(timeline.keys())
        if not frames_sorted:
            continue
        first_f, last_f = frames_sorted[0], frames_sorted[-1]
        i = first_f
        while i <= last_f:
            vis, _ = timeline.get(i, (0.0, None))
            if not (VIS_LOW <= vis < VIS_HIGH):
                i += 1
                continue
            start = i
            while i <= last_f:
                vis_i, _ = timeline.get(i, (0.0, None))
                if not (VIS_LOW <= vis_i < VIS_HIGH):
                    break
                i += 1
            end = i - 1
            length = end - start + 1
            if length >= min_duration:
                episodes.append({"gt_id": gt_id, "start": start, "end": end, "length": length})
    return episodes


def classify_episode_outcome(hyp_sequence: list) -> str:
    """hyp_sequence: [(frame, hyp_id_or_None), ...] -- RETAINED/RECOVERED/FRAGMENTED/PERMANENT_LOSS."""
    non_none = [h for _, h in hyp_sequence if h is not None]
    distinct = set(non_none)
    has_gap = any(h is None for _, h in hyp_sequence)
    if len(distinct) == 0:
        return "PERMANENT_LOSS"
    if len(distinct) == 1:
        return "RETAINED" if not has_gap else "RECOVERED"
    return "FRAGMENTED"


def find_loss_transition_frame(hyp_sequence: list):
    """FRAGMENTED/PERMANENT_LOSS episode'lerde, 'kaybin basladigi' -- ILK --
    kare. Iki ayri gecis sekli var, ikisi de yakalanmali:
      (a) eslesmeden BOSLUGA gecis (None) -- Faz 3.2'nin gercek deseni
          (7 -> None...None -> 40)
      (b) BOSLUKSUZ ani ID degisimi (7 -> 40, arada hic None yok) --
          find_loss_transition_frame'in ILK versiyonu bunu KACIRIYORDU
          (sadece son eslesen karenin index'ini takip edip +1 donuyordu,
          bu da FARKLI id'ye dogrudan gecen -- gap'siz -- durumlarda
          yanlislikla None donduruyordu; birim testle yakalandi)."""
    last_hyp = None
    for frame, h in hyp_sequence:
        if h is None:
            if last_hyp is not None:
                return frame  # ilk "eslesmeden bosluga" gecisi
        elif last_hyp is not None and h != last_hyp:
            return frame  # ilk "boslu(k)suz ani ID degisimi"
        if h is not None:
            last_hyp = h
    return hyp_sequence[0][0] if hyp_sequence else None  # PERMANENT_LOSS: bastan sona None


def classify_transition_reason(vis: float, best_iou: float, best_conf: float, hyp_match_exists: bool) -> str:
    if hyp_match_exists:
        return None
    if best_iou <= 0.0:
        return "P1_NO_DETECTION"
    if best_iou < IOU_MATCH_THRESH:
        return "P2_LOW_IOU"
    if best_conf < TRACK_THRESH:
        return "P3_TRACK_THRESH_GATE"
    return "P4_ASSIGNMENT_CONFLICT"


def _collect_baseline_hyp(seq_dir: Path, fps: float, limit_frames=None) -> dict:
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


def _build_frame_matches(gt_by_frame_full: dict, hyp_by_frame: dict, n_frames: int) -> dict:
    """{frame: {gt_id: hyp_id}} -- HER karede Hungarian."""
    frame_matches = {}
    for frame in range(1, n_frames + 1):
        gt_rows = gt_by_frame_full.get(frame, [])
        hyp_rows = [(tid, tuple(box)) for tid, box in hyp_by_frame.get(frame, [])]
        frame_matches[frame] = _match_gt_to_hyp(gt_rows, hyp_rows)
    return frame_matches


def _episode_hyp_sequence(gt_id: int, start: int, end: int, frame_matches: dict) -> list:
    return [(f, frame_matches.get(f, {}).get(gt_id)) for f in range(start, end + 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--limit-frames", type=int, default=None, help="Hizli duman testi icin")
    args = parser.parse_args()
    cache_path = CACHE_DIR / f"partial_visibility_raw_{args.sequence}.json"

    seq_dir = DATA_DIR / args.sequence
    gt_by_frame_full = _load_gt_all_visibility(seq_dir / "gt" / "gt.txt")
    n_frames = max(gt_by_frame_full.keys())

    episodes = find_partial_visibility_episodes(gt_by_frame_full)
    print(f"=== {args.sequence}: {len(episodes)} partial-visibility episode bulundu "
          f"(0.20<=vis<0.50, >={MIN_PARTIAL_DURATION} kare surekli) ===")
    if not episodes:
        print("  Hic episode yok -- bu sekansta bu hata modu gozlenmiyor.")
        return []

    if args.from_cache:
        if not cache_path.is_file():
            raise SystemExit(f"Cache yok: {cache_path} -- once --from-cache OLMADAN calistirin")
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        episode_hyp_seqs = {int(k): v for k, v in raw["episode_hyp_seqs"].items()}
        transition_dets = {int(k): v for k, v in raw["transition_dets"].items()}
    else:
        print(f"=== hyp uretiliyor (conf={BASELINE_CONF}, track_thresh={TRACK_THRESH}) ===")
        hyp_by_frame = _collect_baseline_hyp(seq_dir, read_seq_fps(seq_dir), args.limit_frames)
        frame_matches = _build_frame_matches(gt_by_frame_full, hyp_by_frame, n_frames)

        episode_hyp_seqs = {}
        transition_frames_needed = set()
        for idx, e in enumerate(episodes):
            seq = _episode_hyp_sequence(e["gt_id"], e["start"], e["end"], frame_matches)
            episode_hyp_seqs[idx] = seq
            outcome = classify_episode_outcome(seq)
            if outcome in ("FRAGMENTED", "PERMANENT_LOSS"):
                tf = find_loss_transition_frame(seq)
                if tf is not None:
                    transition_frames_needed.add(tf)

        print(f"=== {len(transition_frames_needed)} gecis karesinde ham tespit cikariliyor (conf={LOW_CONF}) ===")
        boundary_dets = _collect_boundary_detections(seq_dir, transition_frames_needed) if transition_frames_needed else {}

        transition_dets = {}
        for f in transition_frames_needed:
            transition_dets[f] = [(d.xyxy, d.conf) for d in boundary_dets.get(f, [])]

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "episode_hyp_seqs": {str(k): v for k, v in episode_hyp_seqs.items()},
            "transition_dets": {str(k): v for k, v in transition_dets.items()},
        }), encoding="utf-8")
        print(f"  Cache'e yazildi: {cache_path}")

    # gt_id -> {frame: (vis, box)} -- P1-P4 tanisi icin GT kutusu lazim
    identity_timeline: dict = {}
    for frame, rows in gt_by_frame_full.items():
        for gt_id, box, vis in rows:
            identity_timeline.setdefault(gt_id, {})[frame] = (vis, box)

    rows = []
    for idx, e in enumerate(episodes):
        seq = episode_hyp_seqs[idx]
        outcome = classify_episode_outcome(seq)
        reason = None
        if outcome in ("FRAGMENTED", "PERMANENT_LOSS"):
            tf = find_loss_transition_frame(seq)
            if tf is not None and tf in identity_timeline.get(e["gt_id"], {}):
                vis, box = identity_timeline[e["gt_id"]][tf]
                dets = transition_dets.get(tf, transition_dets.get(str(tf), []))
                if dets:
                    best = max(dets, key=lambda d: iou_xyxy(box, d[0]))
                    best_iou, best_conf = iou_xyxy(box, best[0]), best[1]
                else:
                    best_iou, best_conf = 0.0, 0.0
                hyp_match_exists = False  # tf tanimi geregi (loss transition), her zaman False
                reason = classify_transition_reason(vis, best_iou, best_conf, hyp_match_exists)
        rows.append({**e, "outcome": outcome, "transition_reason": reason})

    print(f"\n{'gt_id':>6s} {'gap':>11s} {'len':>4s} {'outcome':>16s} {'reason':>22s}")
    for r in rows:
        print(f"{r['gt_id']:6d} {r['start']:5d}-{r['end']:<5d} {r['length']:4d} "
              f"{r['outcome']:>16s} {str(r['transition_reason']):>22s}")

    counts = Counter(r["outcome"] for r in rows)
    total = len(rows)
    retained, recovered, fragmented, permanent = (counts.get(k, 0) for k in
                                                    ("RETAINED", "RECOVERED", "FRAGMENTED", "PERMANENT_LOSS"))
    pvtr = (retained + recovered) / total if total else float("nan")
    frag_rate = fragmented / total if total else float("nan")
    print(f"\n=== {args.sequence} ozet ===")
    print(f"  Toplam episode: {total}")
    print(f"  RETAINED: {retained}  RECOVERED: {recovered}  FRAGMENTED: {fragmented}  PERMANENT_LOSS: {permanent}")
    print(f"  PVTR (Partial Visibility Track Retention Rate) = {pvtr:.1%}")
    print(f"  Fragmentation Rate = {frag_rate:.1%}")
    reason_counts = Counter(r["transition_reason"] for r in rows if r["transition_reason"])
    if reason_counts:
        print("  Gecis nedeni dagilimi (FRAGMENTED+PERMANENT_LOSS):")
        for k, v in reason_counts.most_common():
            print(f"    {k}: {v}")

    return rows


if __name__ == "__main__":
    main()
