"""Faz 3.0: occlusion sinirinda detection recall/localizasyon -- YOLO
`conf_threshold` sweep'i.

Faz 2.10'da "association failure" hipotezi elendi (tracker zaten Hungarian
kullaniyordu, ORR evaluator'ini de Hungarian yapmak MOT17-04'te SIFIR
degisiklik yaratti). Geriye kalan aciklama: occlusion sinirinda detection
recall/IoU hassasiyeti. Bu script kullanicinin spesifik istegi uzerine
SADECE `conf_threshold`'u degistirir; ReID checkpoint'i, fps-farkinda
mesafe butcesi, embedding gallery, LostTrack penceresi, Hungarian atama ve
NMS IoU esigi hep SABIT (Faz 2.7/2.8/2.9/2.10'daki "en iyi" -- condition D
-- konfigurasyonu).

**Onemli, ayri bir bulgu (bu script'i yazarken ortaya cikti):**
`AdaptiveTracker.track_thresh` (varsayilan 0.5) `update()` icinde AYRI, IKINCI
bir guven esigi uygular (`dets = [d for d in detections if d.conf >= self.
track_thresh]`) -- YOLO'nun kendi `conf_threshold`'undan BAGIMSIZ. Faz 2.x'in
TUM onceki scriptlerinde bu hic override edilmedi (varsayilan 0.5 kaldi),
yani YOLO conf_threshold=0.35 olarak ayarlanmis olsa bile, tracker 0.35-0.5
arasindaki her tespiti SESSIZCE atiyordu -- efektif esik hep 0.5'ti. Bu
sweep'in 0.5'in ALTINDAKI degerleri (0.30/0.25/0.20/0.15) gercekten test
edebilmesi icin `track_thresh=0.0` set edildi (yani tracker'in kendi ikinci
esigi devre disi, tek esik YOLO'nunki -- kullanicinin "sadece conf_threshold"
disiplinini gercekten uygulamanin dogru yolu budur).

**Performans optimizasyonu (matematiksel olarak tam esdeger, yaklaşıklık
degil):** Ultralytics conf-filtreyi NMS'ten ONCE uygular (bkz.
detection/yolo_detector.py: `model.predict(conf=...)`). Standart greedy
NMS'te bir kutu SADECE kendisinden YUKSEK skorlu, ortusen bir kutu
tarafindan bastirilabilir. Yani skoru >= X olan bir kutu, en dusuk
threshold'da (0.15) NMS'ten sag ciktiysa, threshold=X (X>=0.15) ile TEK
BASINA yeniden calistirilsaydi da AYNEN hayatta kalirdi -- onu bastirabilecek
her kutu zaten kendisinden yuksek skorludur, dolayisiyla X esiginde de zaten
mevcuttur; X ile 0.15 arasindaki (yani X'in disinda kalan, 0.15'in icinde
kalan) kutular hep DAHA DUSUK skorlu olduklarindan hicbirini bastiramaz.
Bu yuzden: tespit+embedding cikarma SADECE conf=0.15'te BIR KEZ yapilir;
daha yuksek esikler o tek pass'in ciktisini (detection, embedding) ciftleri
halinde filtreleyerek elde edilir -- 5 kez YOLO+OSNet calistirmaktan
SONUC olarak farksiz, ama ~5x daha hizli.

Kullanim:
    python -m evaluation.run_confidence_sweep
    python -m evaluation.run_confidence_sweep --limit-frames 40  # hizli duman testi
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from detection.yolo_detector import YoloDetector
from evaluation.mot_metrics import evaluate_sequence, load_mot_gt, read_seq_fps, summarize, xyxy_to_xywh
from evaluation.occlusion_recovery_rate import (
    DEFAULT_CHECKPOINT,
    _load_gt_all_visibility,
    compute_orr,
    frame_level_match_rate,
    summarize_events,
)
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MOT17"
SEQUENCE = "MOT17-04-FRCNN"
CONF_THRESHOLDS = [0.35, 0.30, 0.25, 0.20, 0.15]
LOWEST_CONF = min(CONF_THRESHOLDS)
SEARCH_WINDOW = 30  # Faz 2.9/2.10'da MOT17-04'te tek gecerli sinyalin goruldugu pencere
# Faz 2.7/2.8/2.9/2.10'daki "en iyi" (condition D) konfigurasyonu -- degismiyor.
# track_thresh=0.0: yukaridaki docstring'deki ikinci-esik bulgusu yuzunden.
FIXED_TRACKER_KWARGS = dict(
    weight_mode="adaptive", base_weights=(0.4, 0.3, 0.3), track_thresh=0.0,
    association_dist_ratio=0.08, use_missed_trust=True,
    use_embedding_gallery=True, use_fps_scaling=True,
)


class _PrecomputedExtractor:
    """Ayni karenin onceden (conf=LOWEST_CONF'ta) cikarilmis embedding'lerini
    dondurur -- gercek OSNet'i sweep'in her esiginde tekrar cagirmamak icin.
    `AdaptiveTracker._extract_embeddings`, `extractor.extract(frame, boxes)`'i
    `boxes = [dets[i].xyxy for i in person_idx]` ile, yani `dets`'le AYNI
    sirada cagirir -- bu sinif o sirayla onceden hizalanmis `embeddings`
    listesini oldugu gibi geri verir, frame/boxes'a hic bakmaz."""

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def extract(self, frame, boxes_xyxy):
        return self._embeddings


def _detect_and_embed_once(seq_dir: Path, detector, extractor, limit_frames=None):
    """conf=LOWEST_CONF'ta TEK GECIS: her kare icin (Detection, embedding)
    ciftleri + kare boyutu (frame_diag hesaplamasi icin, gercek piksel
    verisi lazim degil)."""
    img_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    if limit_frames:
        img_paths = img_paths[:limit_frames]
    per_frame = []
    frame_shape = None
    for img_path in img_paths:
        frame = cv2.imread(str(img_path))
        if frame_shape is None:
            frame_shape = frame.shape
        detections = detector.detect(frame)
        person_dets = [d for d in detections if d.cls_name == "person"]
        embeddings = extractor.extract(frame, [d.xyxy for d in person_dets]) if person_dets else []
        per_frame.append(list(zip(person_dets, embeddings)))
    return per_frame, frame_shape


def _run_tracker_at_threshold(per_frame_dets_embs, conf_threshold: float, fps: float, dummy_frame) -> dict:
    tracker = AdaptiveTracker(fps=fps, **FIXED_TRACKER_KWARGS)
    hyp_by_frame_xyxy = {}
    for i, pairs in enumerate(per_frame_dets_embs, start=1):
        kept = [(d, e) for d, e in pairs if d.conf >= conf_threshold]
        dets = [d for d, _ in kept]
        embs = [e for _, e in kept]
        tracks = tracker.update(dets, dummy_frame, extractor=_PrecomputedExtractor(embs))
        hyp_by_frame_xyxy[i] = [(t.track_id, t.xyxy) for t in tracks]
    return hyp_by_frame_xyxy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=SEQUENCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--limit-frames", type=int, default=None, help="Hizli duman testi icin ilk N kareyle sinirla")
    args = parser.parse_args()

    seq_dir = DATA_DIR / args.sequence
    gt_by_frame = load_mot_gt(seq_dir / "gt" / "gt.txt")
    gt_by_frame_full = _load_gt_all_visibility(seq_dir / "gt" / "gt.txt")
    seq_fps = read_seq_fps(seq_dir)

    detector = YoloDetector(weights="yolov8n.pt", conf_threshold=LOWEST_CONF, classes=[0])
    checkpoint_path = str(args.checkpoint) if args.checkpoint.is_file() else None
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=checkpoint_path)

    print(f"=== {args.sequence} ({seq_fps:.0f} FPS) -- tek gecis, conf={LOWEST_CONF} ===")
    per_frame_dets_embs, frame_shape = _detect_and_embed_once(seq_dir, detector, extractor, args.limit_frames)
    dummy_frame = np.zeros(frame_shape, dtype=np.uint8)
    total_dets = sum(len(p) for p in per_frame_dets_embs)
    print(f"  {len(per_frame_dets_embs)} kare, {total_dets} toplam tespit (conf>={LOWEST_CONF})")

    rows = []
    for conf in CONF_THRESHOLDS:
        hyp_by_frame_xyxy = _run_tracker_at_threshold(per_frame_dets_embs, conf, seq_fps, dummy_frame)

        hyp_by_frame_xywh = {
            f: [(tid, xyxy_to_xywh(box)) for tid, box in fr_rows] for f, fr_rows in hyp_by_frame_xyxy.items()
        }
        acc = evaluate_sequence(gt_by_frame, hyp_by_frame_xywh)
        summary = summarize({"row": acc})
        r = summary.loc["row"]

        n_visible, n_matched, match_rate = frame_level_match_rate(gt_by_frame_full, hyp_by_frame_xyxy)
        events = compute_orr(gt_by_frame_full, hyp_by_frame_xyxy, min_occlusion_frames=5,
                              max_recovery_gap=60, search_window=SEARCH_WINDOW)
        within = [e for e in events if e["within_window"]]
        orr_summary = summarize_events(within, f"conf={conf}")

        row = {
            "conf": conf, "n_visible": n_visible, "n_matched": n_matched, "match_rate": match_rate,
            "n_events": orr_summary["n_events"], "n_valid": orr_summary["n_valid"],
            "n_recovered": orr_summary["n_recovered"], "orr": orr_summary["orr"],
            "mota": float(r["mota"]), "idf1": float(r["idf1"]), "idsw": int(r["num_switches"]),
            "fp": int(r["num_false_positives"]), "fn": int(r["num_misses"]),
        }
        rows.append(row)
        orr_str = f"{row['orr']:.1%}" if row["n_valid"] else "n/a (gecerli event yok)"
        print(f"\n=== conf={conf} ===")
        print(f"  Frame match rate: {n_matched}/{n_visible} ({match_rate:.1%})")
        print(f"  ORR (window={SEARCH_WINDOW}): {row['n_recovered']}/{row['n_valid']} "
              f"(/{row['n_events']} event) -> {orr_str}")
        print(f"  MOTA={row['mota']:.3f} IDF1={row['idf1']:.3f} IDSW={row['idsw']} FP={row['fp']} FN={row['fn']}")

    print(f"\n\n{'Conf':>5s} {'Match%':>8s} {'ValidEv':>9s} {'ORR':>9s} {'MOTA':>7s} {'IDF1':>7s} {'IDSW':>5s} {'FP':>6s} {'FN':>7s}")
    for row in rows:
        orr_str = f"{row['orr']:.1%}" if row["n_valid"] else "n/a"
        valid_str = f"{row['n_valid']}/{row['n_events']}"
        print(f"{row['conf']:5.2f} {row['match_rate']:7.1%} {valid_str:>9s} {orr_str:>9s} "
              f"{row['mota']:7.3f} {row['idf1']:7.3f} {row['idsw']:5d} {row['fp']:6d} {row['fn']:7d}")

    return rows


if __name__ == "__main__":
    main()
