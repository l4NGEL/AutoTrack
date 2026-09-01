"""Faz 3.3'un actigi soruya kucuk, izole bir prototip -- kullanicinin
istegiyle, `tracking/adaptive_tracker.py`'ye (AutoTrack'in asil, cok
test edilmis kodu) HIC DOKUNMADAN.

Somut, kod-seviyesinde dogrulanmis gozlem: `AdaptiveTracker._reid_score`
(PRIMARY cost matrix, HER karede kullanilir) SADECE `track.embedding`'i
(tek, EMA ile harmanlanmis gorunum) kullaniyor. `embedding_gallery`
(Faz 2.5 -- birden fazla BELIRGIN FARKLI gecmis gorunum) SADECE
`_try_revive`'de (kayip-track kurtarma) kontrol ediliyor, aktif takip
sirasindaki PRIMARY eslesmede HIC degil.

Hipotez: partial-visibility (0.20<=vis<0.50) sirasinda kisinin GORUNUMU
(aci, kismi kirpim) degisebilir; tek EMA-blended embedding bunu iyi
temsil etmeyebilir ama galerideki (farkli acilardan) gecmis gorunumlerden
biri daha iyi eslesebilir.

`GalleryAwareTracker`, `AdaptiveTracker`'in bir ALT SINIFI -- SADECE
`_reid_score`'u override eder (base EMA skoru İLE galerideki EN İYİ
benzerlik'in MAKSIMUMUNU alir), geri kalan HER SEY (Hungarian, mesafe
siniri, fps-farkindaligi, missed-trust, vb.) BIREBIR AYNI.

**Onemli, ONCEDEN belirtilmesi gereken sinir:** P1 (hic tespit yok, Faz
3.3'un basarisiz episode'lerin %48.7'si) bu fikirle DUZELMEZ -- cost
matrix sadece VAR OLAN tespitler arasinda calisir. Bu prototip SADECE
P2/P3/P4'u (detection var ama eslesmeyi kaybediyor -- basarisiz
episode'lerin en fazla %51.3'u) hedefleyebilir.

Kullanim:
    python -m evaluation.prototype_gallery_reid --sequence MOT17-04-FRCNN
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2

from detection.yolo_detector import YoloDetector
from evaluation.mot_metrics import read_seq_fps
from evaluation.occlusion_recovery_rate import DEFAULT_CHECKPOINT, _load_gt_all_visibility
from evaluation.partial_visibility_forensics import (
    BASELINE_CONF,
    TRACK_THRESH,
    _build_frame_matches,
    _episode_hyp_sequence,
    classify_episode_outcome,
    find_partial_visibility_episodes,
)
from reid.feature_extractor import OSNetFeatureExtractor, cosine_similarity
from tracking.adaptive_tracker import AdaptiveTracker

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MOT17"
DEFAULT_SEQUENCE = "MOT17-04-FRCNN"


class GalleryAwareTracker(AdaptiveTracker):
    """SADECE _reid_score'u override eder. AdaptiveTracker'in KENDI kodu
    degismedi -- bu tamamen ayri bir sinif, sadece miras aliyor."""

    def _reid_score(self, track, det_embedding) -> float:
        base = super()._reid_score(track, det_embedding)
        if not self.use_embedding_gallery or not track.embedding_gallery or det_embedding is None:
            return base
        gallery_sims = [
            max(0.0, (cosine_similarity(g, det_embedding) + 1.0) / 2.0)
            for g in track.embedding_gallery
        ]
        return max(base, max(gallery_sims, default=0.0))


def _collect_hyp(seq_dir: Path, fps: float, tracker_cls) -> dict:
    detector = YoloDetector(weights="yolov8n.pt", conf_threshold=BASELINE_CONF, classes=[0])
    checkpoint_path = str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.is_file() else None
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=checkpoint_path)
    tracker = tracker_cls(weight_mode="adaptive", base_weights=(0.4, 0.3, 0.3),
                           track_thresh=TRACK_THRESH, fps=fps)

    img_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    hyp_by_frame = {}
    for i, img_path in enumerate(img_paths, start=1):
        frame = cv2.imread(str(img_path))
        detections = detector.detect(frame)
        person_dets = [d for d in detections if d.cls_name == "person"]
        tracks = tracker.update(person_dets, frame, extractor=extractor)
        hyp_by_frame[i] = [(t.track_id, list(t.xyxy)) for t in tracks]
    return hyp_by_frame


def _evaluate(label: str, gt_by_frame_full: dict, episodes: list, hyp_by_frame: dict, n_frames: int) -> dict:
    frame_matches = _build_frame_matches(gt_by_frame_full, hyp_by_frame, n_frames)
    outcomes = []
    for e in episodes:
        seq = _episode_hyp_sequence(e["gt_id"], e["start"], e["end"], frame_matches)
        outcomes.append(classify_episode_outcome(seq))
    counts = Counter(outcomes)
    total = len(outcomes)
    retained, recovered, fragmented, permanent = (counts.get(k, 0) for k in
                                                    ("RETAINED", "RECOVERED", "FRAGMENTED", "PERMANENT_LOSS"))
    pvtr = (retained + recovered) / total if total else float("nan")
    frag_rate = fragmented / total if total else float("nan")
    print(f"\n=== {label} ===")
    print(f"  RETAINED={retained} RECOVERED={recovered} FRAGMENTED={fragmented} PERMANENT_LOSS={permanent}")
    print(f"  PVTR={pvtr:.1%}  FragRate={frag_rate:.1%}")
    return {"label": label, "retained": retained, "recovered": recovered, "fragmented": fragmented,
            "permanent_loss": permanent, "total": total, "pvtr": pvtr, "frag_rate": frag_rate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    args = parser.parse_args()

    seq_dir = DATA_DIR / args.sequence
    gt_by_frame_full = _load_gt_all_visibility(seq_dir / "gt" / "gt.txt")
    n_frames = max(gt_by_frame_full.keys())
    seq_fps = read_seq_fps(seq_dir)

    episodes = find_partial_visibility_episodes(gt_by_frame_full)
    print(f"=== {args.sequence}: {len(episodes)} partial-visibility episode "
          f"(baseline'daki AYNI episode'ler, GT'den geliyor -- degismiyor) ===")
    if not episodes:
        print("  Hic episode yok, karsilastirma yapilamaz.")
        return

    print("\n=== BASELINE (AdaptiveTracker, degismedi) hyp uretiliyor ===")
    baseline_hyp = _collect_hyp(seq_dir, seq_fps, AdaptiveTracker)
    print("\n=== PROTOTIP (GalleryAwareTracker) hyp uretiliyor ===")
    prototype_hyp = _collect_hyp(seq_dir, seq_fps, GalleryAwareTracker)

    baseline_result = _evaluate("BASELINE (AdaptiveTracker)", gt_by_frame_full, episodes, baseline_hyp, n_frames)
    prototype_result = _evaluate("PROTOTIP (GalleryAwareTracker: galeri PRIMARY eslesmede de kullanilir)",
                                  gt_by_frame_full, episodes, prototype_hyp, n_frames)

    print(f"\n=== Karsilastirma ({args.sequence}) ===")
    print(f"  PVTR:      {baseline_result['pvtr']:.1%} -> {prototype_result['pvtr']:.1%}")
    print(f"  FragRate:  {baseline_result['frag_rate']:.1%} -> {prototype_result['frag_rate']:.1%}")
    print(f"  PERMANENT_LOSS: {baseline_result['permanent_loss']} -> {prototype_result['permanent_loss']}")
    print(f"  (Hatirlatma: P1/hic-tespit-yok episode'leri bu fikirle prensipte DUZELEMEZ -- "
          f"sadece detection VAR olan ama eslesmeyi kaybeden durumlar etkilenebilir.)")

    return baseline_result, prototype_result


if __name__ == "__main__":
    main()
