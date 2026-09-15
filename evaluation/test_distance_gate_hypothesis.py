"""Faz 3.4'un actik biraktigi soruyu DOGRUDAN test eder: partial-visibility
rejiminde basarisiz olan P2_LOW_IOU / P4_ASSIGNMENT_CONFLICT episode'lari
gercekten `association_dist_ratio` (max_dist mesafe siniri) yuzunden mi
kaybediliyor?

Faz 3.4 (GalleryAwareTracker prototipi, prototype_gallery_reid.py) _reid_score'u
degistirmenin BIT-BIT SIFIR etki yarattigini bulmustu; P1+P3 (%65.7) zaten
cost matrix'e hic girmiyordu (yapisal olarak duzelemez), ama geriye kalan
P2+P4 (%34.3) icin de sifir degisiklik olmasi, mesafe siniri adaylari ReID
skoru hesaplanmadan ONCE eledigi cikarimina goturmustu -- bu cikarim o
oturumda AYRICA TEST EDILMEMISTI, sadece sonuctan cikarilan en olasi
aciklamaydi. Bu script onu test ediyor.

Yontem: ayni 38 partial-visibility episode'unu (partial_visibility_forensics.py
ile AYNI taksonomi/esik/fonksiyonlar, kod TEKRARLANMADI) association_dist_ratio
icin iki degerle yeniden siniflandirir:
  0.08 -- guncel varsayilan (Faz 2.4'un sikilastirdigi deger)
  0.15 -- Faz 2.4 ONCESI deger (Faz 2.1'in koydugu ilk sinir)

BILEREK riskli bir deger test ediliyor: 0.15 tam olarak Faz 2.4'un
sikilastirdigi deger -- o fazda GERCEK, dogrulanmis yanlis-birlestirmeler
(685px ve 309px'lik tek-kare sicramalar) bu yuzden bulunmustu. Yani bu
script'in cevaplamasi gereken soru TEK BOYUTLU degil:
  (a) 0.15, herhangi bir P2/P4 episode'unu kurtariyor mu?
  (b) kurtariyorsa, ayni zamanda Faz 2.4/2.6'nin bulduguyla AYNI TURDEN
      imkansiz ardisik-kare sicramalarini geri getiriyor mu (yeniden,
      taze bir tarama ile -- eski bulgu tekrar dogrulanmis SAYILMAZ)?

tracking/adaptive_tracker.py'ye HICBIR DOKUNUS yok -- association_dist_ratio
zaten disaridan gecirilebilir bir __init__ parametresi (bkz. run_isolation_ablation.py'de
ayni parametre, farkli amacla, zaten kullaniliyor).

Kullanim:
    python -m evaluation.test_distance_gate_hypothesis
    python -m evaluation.test_distance_gate_hypothesis --sequence MOT17-04-FRCNN --from-cache
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from detection.yolo_detector import YoloDetector
from evaluation.mot_metrics import read_seq_fps
from evaluation.occlusion_event_forensics import _collect_boundary_detections
from evaluation.occlusion_recovery_rate import DEFAULT_CHECKPOINT, _load_gt_all_visibility
from evaluation.partial_visibility_forensics import (
    DATA_DIR,
    DEFAULT_SEQUENCE,
    BASELINE_CONF,
    LOW_CONF,
    TRACK_THRESH,
    _build_frame_matches,
    _episode_hyp_sequence,
    classify_episode_outcome,
    classify_transition_reason,
    find_loss_transition_frame,
    find_partial_visibility_episodes,
)
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker
from tracking.bytetrack_wrapper import iou as iou_xyxy

CACHE_DIR = Path(__file__).resolve().parent / "_cache"
RATIOS_TO_TEST = [0.08, 0.15]  # guncel (Faz 2.4) vs Faz 2.4-oncesi


def _collect_hyp(seq_dir: Path, fps: float, association_dist_ratio: float,
                  detector: YoloDetector, extractor, limit_frames=None) -> dict:
    """partial_visibility_forensics._collect_baseline_hyp ile AYNI, tek fark:
    association_dist_ratio disaridan parametre -- detector/extractor da
    disaridan aliniyor ki iki ratio arasinda modelleri yeniden yuklemeyelim."""
    tracker = AdaptiveTracker(weight_mode="adaptive", base_weights=(0.4, 0.3, 0.3),
                               track_thresh=TRACK_THRESH, fps=fps,
                               association_dist_ratio=association_dist_ratio)
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


def _max_consecutive_jump(hyp_by_frame: dict) -> dict:
    """Her track_id icin ARDISIK kare (i, i+1) ciftlerinde merkez sicramasini
    olcer, TUM sekanstaki en buyugunu doner. Faz 2.4/2.6'daki "ayni track_id,
    tek karede imkansiz sicrama" taramasinin (o zaman tek seferlik/tests
    disi bir betikti) tekrar kullanilabilir, test edilebilir hali."""
    positions: dict = {}
    for frame, entries in hyp_by_frame.items():
        for tid, box in entries:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            positions.setdefault(tid, {})[frame] = (cx, cy)

    worst = {"track_id": None, "frame": None, "jump_px": 0.0}
    for tid, frames in positions.items():
        sorted_frames = sorted(frames.keys())
        for a, b in zip(sorted_frames, sorted_frames[1:]):
            if b != a + 1:
                continue  # sadece GERCEKTEN ardisik kare ciftleri sayilir
            (x1, y1), (x2, y2) = frames[a], frames[b]
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if dist > worst["jump_px"]:
                worst = {"track_id": tid, "frame": b, "jump_px": dist}
    return worst


def _classify_all_episodes(episodes, gt_by_frame_full, hyp_by_frame, n_frames, seq_dir):
    """Bir tek ratio'nun hyp_by_frame'inden episode outcome + (basarisizsa)
    P1-P4 nedenini cikarir. partial_visibility_forensics.main()'in ayni
    mantigi, tek-ratio icin yeniden kullanilabilir hale getirilmis hali."""
    frame_matches = _build_frame_matches(gt_by_frame_full, hyp_by_frame, n_frames)

    episode_hyp_seqs, transition_frames_needed = {}, set()
    for idx, e in enumerate(episodes):
        seq = _episode_hyp_sequence(e["gt_id"], e["start"], e["end"], frame_matches)
        episode_hyp_seqs[idx] = seq
        outcome = classify_episode_outcome(seq)
        if outcome in ("FRAGMENTED", "PERMANENT_LOSS"):
            tf = find_loss_transition_frame(seq)
            if tf is not None:
                transition_frames_needed.add(tf)

    boundary_dets = _collect_boundary_detections(seq_dir, transition_frames_needed) if transition_frames_needed else {}
    transition_dets = {f: [(d.xyxy, d.conf) for d in boundary_dets.get(f, [])] for f in transition_frames_needed}

    identity_timeline: dict = {}
    for frame, rows in gt_by_frame_full.items():
        for gt_id, box, vis in rows:
            identity_timeline.setdefault(gt_id, {})[frame] = (vis, box)

    results = []
    for idx, e in enumerate(episodes):
        seq = episode_hyp_seqs[idx]
        outcome = classify_episode_outcome(seq)
        reason = None
        if outcome in ("FRAGMENTED", "PERMANENT_LOSS"):
            tf = find_loss_transition_frame(seq)
            if tf is not None and tf in identity_timeline.get(e["gt_id"], {}):
                vis, box = identity_timeline[e["gt_id"]][tf]
                dets = transition_dets.get(tf, [])
                if dets:
                    best = max(dets, key=lambda d: iou_xyxy(box, d[0]))
                    best_iou, best_conf = iou_xyxy(box, best[0]), best[1]
                else:
                    best_iou, best_conf = 0.0, 0.0
                reason = classify_transition_reason(vis, best_iou, best_conf, False)
        results.append({**e, "outcome": outcome, "reason": reason})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    seq_dir = args.data_dir / args.sequence
    gt_by_frame_full = _load_gt_all_visibility(seq_dir / "gt" / "gt.txt")
    n_frames = max(gt_by_frame_full.keys())
    fps = read_seq_fps(seq_dir)

    episodes = find_partial_visibility_episodes(gt_by_frame_full)
    print(f"=== {args.sequence}: {len(episodes)} partial-visibility episode ===")
    if not episodes:
        print("  Hic episode yok -- bu sekansta test edilecek bir sey yok.")
        return

    detector = extractor = None
    if not args.from_cache:
        detector = YoloDetector(weights="yolov8n.pt", conf_threshold=BASELINE_CONF, classes=[0])
        checkpoint_path = str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.is_file() else None
        extractor = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=checkpoint_path)

    per_ratio = {}
    for ratio in RATIOS_TO_TEST:
        cache_path = CACHE_DIR / f"distance_gate_hyp_{args.sequence}_{ratio}.json"
        if args.from_cache:
            if not cache_path.is_file():
                raise SystemExit(f"Cache yok: {cache_path} -- once --from-cache OLMADAN calistirin")
            hyp_by_frame = {int(k): v for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()}
        else:
            print(f"\n=== association_dist_ratio={ratio} ile hyp uretiliyor ===")
            hyp_by_frame = _collect_hyp(seq_dir, fps, ratio, detector, extractor, args.limit_frames)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({str(k): v for k, v in hyp_by_frame.items()}), encoding="utf-8")
            print(f"  Cache'e yazildi: {cache_path}")

        results = _classify_all_episodes(episodes, gt_by_frame_full, hyp_by_frame, n_frames, seq_dir)
        jump = _max_consecutive_jump(hyp_by_frame)
        per_ratio[ratio] = {"results": results, "jump": jump}
        print(f"  En buyuk ardisik-kare sicramasi: {jump['jump_px']:.0f}px "
              f"(track {jump['track_id']}, kare {jump['frame']})")

    base_ratio, wide_ratio = RATIOS_TO_TEST[0], RATIOS_TO_TEST[-1]
    base_results = per_ratio[base_ratio]["results"]
    wide_results = per_ratio[wide_ratio]["results"]

    print(f"\n{'gt_id':>6s} {'gap':>11s} {f'outcome@{base_ratio}':>20s} {f'reason@{base_ratio}':>18s} "
          f"{f'outcome@{wide_ratio}':>20s} {f'reason@{wide_ratio}':>18s} {'DEGISTI':>8s}")
    n_flipped, n_p2p4_flipped = 0, 0
    for b, w in zip(base_results, wide_results):
        assert b["gt_id"] == w["gt_id"] and b["start"] == w["start"]
        changed = b["outcome"] != w["outcome"]
        n_flipped += changed
        was_p2p4 = b["reason"] in ("P2_LOW_IOU", "P4_ASSIGNMENT_CONFLICT")
        if changed and was_p2p4:
            n_p2p4_flipped += 1
        print(f"{b['gt_id']:6d} {b['start']:5d}-{b['end']:<5d} "
              f"{b['outcome']:>20s} {str(b['reason']):>18s} "
              f"{w['outcome']:>20s} {str(w['reason']):>18s} {'EVET' if changed else '':>8s}")

    print(f"\n=== SONUC ({args.sequence}) ===")
    print(f"  {n_flipped}/{len(base_results)} episode outcome degisti (ratio {base_ratio} -> {wide_ratio})")
    print(f"  Bunlarin {n_p2p4_flipped}'i P2/P4 (teorik olarak mesafe-siniri-kaynakli) idi")
    print(f"  En buyuk ardisik-kare sicrama: {base_ratio}={per_ratio[base_ratio]['jump']['jump_px']:.0f}px  "
          f"{wide_ratio}={per_ratio[wide_ratio]['jump']['jump_px']:.0f}px")
    if per_ratio[wide_ratio]["jump"]["jump_px"] > per_ratio[base_ratio]["jump"]["jump_px"] * 1.5:
        print("  UYARI: genis ratio, Faz 2.4'un duzelttigi turden imkansiz sicramalari "
              "yeniden ortaya cikarmis gorunuyor.")

    return per_ratio


if __name__ == "__main__":
    main()
