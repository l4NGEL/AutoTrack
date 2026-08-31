"""Faz 3.1: Occlusion Event Forensics -- Faz 2.9/2.10/3.0'da "olculemeyen"
bulunan 23 occlusion event'inin (MOT17-04, pencere-ici) HER BIRINI tek tek
inceler, neden basarisiz olduklarini A/B/C/D/E kategorilerine ayirir.

Faz 3.0'in bulgusu: conf_threshold'u 0.35'ten 0.15'e indirmek genel frame
match rate'i %41.6->%51.5 yukseltti ama bu 23 event'in gecerli sayisi 1'de
sabit kaldi -- yani "recall dusuk" aciklamasi zayif. Bu script "neden?"
sorusunu event-bazinda, sayisallastirarak cevaplar.

Kategoriler (once/sonra siniri icin AYRI degerlendirilir, oncelik sirasi
asagida -- bir taraf basarisizsa bu sirayla kontrol edilir):
  E FULL_OCCLUSION_OR_BOUNDARY -- o sinir karesindeki gorunurluk zaten
    cok dusuk (<0.3, yani vis>=0.2 esiginin hemen ustunde, "gorunur"
    sayilmasi kendi basina tartismali)
  A NO_DETECTION      -- o karede (conf>=0.15) GT kutusuna en ufak
                          ortusen (IoU>0) bile hicbir tespit yok
  B LOW_IOU            -- tespit var ama en iyisi IoU<0.5
  C ASSIGNMENT_CONFLICT -- IoU>=0.5'lik bir tespit var AMA o TAM karede
                          Hungarian eslestirmesi (motmetrics'le ayni
                          prensip, bkz. occlusion_recovery_rate.py) bu
                          GT kimligini BASKA bir hyp'e degil, hic
                          eslestirmemis (baska bir GT kimlik kapmis)
  D TRACK_ID_UNSTABLE  -- (SADECE "gecerli" tek event icin) once/sonra
                          ikisi de eslesti ama FARKLI hyp ID -- gercek
                          ID-switch

Bir event'in iki tarafi (once/sonra) farkli kategorilere dusebilir; event'in
GENEL etiketi E > A > B > C oncelik sirasiyla (en temel/kok sorunu
yansitan) -- yani herhangi bir taraf E ise event E, degilse herhangi bir
taraf A ise event A, vs.

Kullanim:
    python -m evaluation.occlusion_event_forensics
"""
import argparse
import json
from pathlib import Path

import cv2

from detection.yolo_detector import YoloDetector
from evaluation.mot_metrics import read_seq_fps
from evaluation.occlusion_recovery_rate import (
    DEFAULT_CHECKPOINT,
    IOU_MATCH_THRESH,
    _load_gt_all_visibility,
    _match_gt_to_hyp,
    compute_orr,
    summarize_events,
)
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker
from tracking.bytetrack_wrapper import iou as iou_xyxy

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MOT17"
CACHE_DIR = Path(__file__).resolve().parent / "_cache"
DEFAULT_SEQUENCE = "MOT17-04-FRCNN"
# Faz 2.9/2.10'un OLCTUGU ayni "resmi" baseline -- YOLO conf=0.35, tracker
# track_thresh VARSAYILANI (Faz 3.0'in bulgusuyla efektif 0.5). Bu forensics
# o spesifik event'leri aciklamali, Faz 3.0'in track_thresh=0.0 varyantini
# DEGIL (kasitli -- "hangi event'ler neden olculemedi" sorusu SIRASIYLA bu
# baseline'a gore soruldu).
# Faz 3.1'in MOT17-05 replikasyonu (kullanicinin istegi): asagidaki
# TUM parametreler ve siniflandirma mantigi MOT17-04'te kullanilanla
# BIREBIR AYNI -- "sonuca gore esik ayarlama" riskinden kacinmak icin
# bilerek DEGISTIRILMEDI. Tek degisen sey --sequence.
BASELINE_CONF = 0.35
LOW_CONF = 0.15  # "hic tespit yok mu" sorusu icin en comert esik (Faz 3.0'la tutarli)
MIN_OCCLUSION_FRAMES = 5
MAX_RECOVERY_GAP = 60
SEARCH_WINDOW = 30
VIS_MARGIN_THRESH = 0.3  # bu esigin altinda "sinir karesi zaten belirsiz gorunur" (kategori E)


def _crowd_stats(gt_by_frame_full: dict, frame: int, gt_id: int, box) -> tuple:
    """(crowd_count, nearest_other_iou) -- ayni karede, ayni sekilde gorunur
    (vis>=0.2) BASKA kac GT kimlik var ve bu kutuyla en yuksek IoU'lu
    olaninki ne kadar (kalabalik/ortusme olcusu)."""
    others = [(oid, obox, ovis) for oid, obox, ovis in gt_by_frame_full.get(frame, [])
              if oid != gt_id and ovis >= 0.2]
    crowd_count = len(others)
    nearest_iou = max((iou_xyxy(box, obox) for _, obox, _ in others), default=0.0)
    return crowd_count, nearest_iou


def _best_detection_at(detections_by_frame: dict, frame: int, gt_box) -> tuple:
    """(best_iou, best_conf) -- o karede (conf>=LOW_CONF) TUM tespitler
    arasinda gt_box'a en yuksek IoU'yu veren."""
    dets = detections_by_frame.get(frame, [])
    if not dets:
        return 0.0, 0.0
    best = max(dets, key=lambda d: iou_xyxy(gt_box, d.xyxy))
    return iou_xyxy(gt_box, best.xyxy), best.conf


def _hyp_match_exists_at(gt_by_frame_full: dict, hyp_by_frame_xyxy: dict, frame: int, gt_id: int) -> bool:
    """O TAM karede, Hungarian GT<->hyp eslestirmesinde bu gt_id'nin bir
    hyp'e eslesip eslesmedigi (arama penceresi YOK, sadece bu kare)."""
    matches = _match_gt_to_hyp(gt_by_frame_full.get(frame, []), hyp_by_frame_xyxy.get(frame, []))
    return gt_id in matches


def _classify_side(vis: float, best_iou: float, hyp_match_exists: bool) -> str | None:
    """Basarili taraf icin None, basarisizsa E/A/B/C dondurur."""
    if hyp_match_exists:
        return None
    if vis < VIS_MARGIN_THRESH:
        return "E"
    if best_iou <= 0.0:
        return "A"
    if best_iou < IOU_MATCH_THRESH:
        return "B"
    return "C"  # best_iou >= esik ama bu TAM karede baska kimlige kapilmis


def classify_event(reason_before, reason_after, before_hyp, after_hyp) -> str:
    """Genel event etiketi -- oncelik E > A > B > C > D > basarili.

    ONEMLI: basari/basarisizlik ONCELIKLE compute_orr'un (search_window ile,
    yani sadece TAM sinir karesini degil, birkac kare geri/ileriyi de tarayan)
    bulmus oldugu before_hyp/after_hyp'e gore belirlenir. reason_before/after
    (bu script'in kendi, SADECE tam sinir karesine bakan tanisi) bir tarafin
    hyp'i GERCEKTEN None ise devreye girer -- yoksa search_window'un
    kurtardigi bir tarafi, o TAM karede gorunurluk dusuk/tespit zayif diye
    yanlislikla "basarisiz" gostermis oluruz (bu fonksiyonun ilk halinde
    tam da bu hataya dusuldu -- bkz. README "Faz 3.1")."""
    if before_hyp is not None and after_hyp is not None:
        return "RECOVERED" if before_hyp == after_hyp else "D"
    reasons = set()
    if before_hyp is None and reason_before is not None:
        reasons.add(reason_before)
    if after_hyp is None and reason_after is not None:
        reasons.add(reason_after)
    for priority in ("E", "A", "B", "C"):
        if priority in reasons:
            return priority
    return "UNKNOWN"  # tutarsizlik -- manuel incelenmeli


def _collect_baseline_hyp(seq_dir: Path, fps: float) -> dict:
    detector = YoloDetector(weights="yolov8n.pt", conf_threshold=BASELINE_CONF, classes=[0])
    checkpoint_path = str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.is_file() else None
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=checkpoint_path)
    tracker = AdaptiveTracker(weight_mode="adaptive", base_weights=(0.4, 0.3, 0.3), fps=fps)  # track_thresh VARSAYILAN

    img_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    hyp_by_frame_xyxy = {}
    for i, img_path in enumerate(img_paths, start=1):
        frame = cv2.imread(str(img_path))
        detections = detector.detect(frame)
        person_dets = [d for d in detections if d.cls_name == "person"]
        tracks = tracker.update(person_dets, frame, extractor=extractor)
        hyp_by_frame_xyxy[i] = [(t.track_id, t.xyxy) for t in tracks]
    return hyp_by_frame_xyxy


def _collect_boundary_detections(seq_dir: Path, frames_needed: set) -> dict:
    """SADECE ihtiyac duyulan (sinir) karelerde, LOW_CONF ile HAM tespit --
    tam sekans taramasi gerekmez (stateless), bu yuzden hizli."""
    detector = YoloDetector(weights="yolov8n.pt", conf_threshold=LOW_CONF, classes=[0])
    img_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    out = {}
    for frame_no in sorted(frames_needed):
        img_path = img_paths[frame_no - 1]
        frame = cv2.imread(str(img_path))
        detections = detector.detect(frame)
        out[frame_no] = [d for d in detections if d.cls_name == "person"]
    return out


def _collect_raw_features(within: list, gt_by_frame_full: dict, hyp_by_frame_xyxy: dict, boundary_dets: dict) -> list:
    """Siniflandirmadan BAGIMSIZ, ham olcumler -- pahali (detection/tracking)
    kismi bu fonksiyondan sonra tekrar calistirmadan, siniflandirma
    mantigini (classify_side/classify_event) DUZELTIP tekrar denemek icin
    ayri tutuldu (bkz. --from-cache)."""
    identity_timeline: dict = {}
    for frame, rows in gt_by_frame_full.items():
        for gt_id, box, vis in rows:
            identity_timeline.setdefault(gt_id, {})[frame] = (vis, box)

    raw_rows = []
    for e in within:
        gt_id, bf, af = e["gt_id"], e["start"] - 1, e["end"] + 1
        vis_before, box_before = identity_timeline[gt_id][bf]
        vis_after, box_after = identity_timeline[gt_id][af]

        crowd_before, nearest_iou_before = _crowd_stats(gt_by_frame_full, bf, gt_id, box_before)
        crowd_after, nearest_iou_after = _crowd_stats(gt_by_frame_full, af, gt_id, box_after)

        best_iou_before, best_conf_before = _best_detection_at(boundary_dets, bf, box_before)
        best_iou_after, best_conf_after = _best_detection_at(boundary_dets, af, box_after)

        hyp_exists_before = _hyp_match_exists_at(gt_by_frame_full, hyp_by_frame_xyxy, bf, gt_id)
        hyp_exists_after = _hyp_match_exists_at(gt_by_frame_full, hyp_by_frame_xyxy, af, gt_id)

        area_before = (box_before[2] - box_before[0]) * (box_before[3] - box_before[1])
        area_after = (box_after[2] - box_after[0]) * (box_after[3] - box_after[1])

        raw_rows.append({
            "gt_id": gt_id, "start": e["start"], "end": e["end"], "length": e["length"],
            "before_hyp": e["before_hyp"], "after_hyp": e["after_hyp"],
            "vis_before": vis_before, "vis_after": vis_after,
            "area_before": area_before, "area_after": area_after,
            "crowd_before": crowd_before, "crowd_after": crowd_after,
            "nearest_iou_before": nearest_iou_before, "nearest_iou_after": nearest_iou_after,
            "best_iou_before": best_iou_before, "best_iou_after": best_iou_after,
            "best_conf_before": best_conf_before, "best_conf_after": best_conf_after,
            "hyp_exists_before": hyp_exists_before, "hyp_exists_after": hyp_exists_after,
        })
    return raw_rows


def _classify_raw_rows(raw_rows: list) -> list:
    """Ham olculere (pahali kisim) SINIFLANDIRMA mantigini (ucuz, hizli)
    uygular -- --from-cache ile bu fonksiyon TEK BASINA tekrar calistirilabilir."""
    rows = []
    for r in raw_rows:
        reason_before = _classify_side(r["vis_before"], r["best_iou_before"], r["hyp_exists_before"])
        reason_after = _classify_side(r["vis_after"], r["best_iou_after"], r["hyp_exists_after"])
        label = classify_event(reason_before, reason_after, r["before_hyp"], r["after_hyp"])
        rows.append({**r, "reason_before": reason_before, "reason_after": reason_after, "label": label})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE,
                         help="Faz 3.1'in MOT17-04'te kullandigi TUM parametreler/siniflandirma mantigi "
                              "degistirilmeden, sadece hangi sekansta calistirilacagi (Faz 3.1 replikasyonu)")
    parser.add_argument("--from-cache", action="store_true",
                         help="Pahali detection/tracking pass'ini atla, bu sekans icin onceki calistirmadan "
                              "kaydedilen ham olculeri kullan (SADECE siniflandirma mantigini test/duzelt icin)")
    args = parser.parse_args()
    cache_path = CACHE_DIR / f"occlusion_event_forensics_raw_{args.sequence}.json"

    if args.from_cache:
        if not cache_path.is_file():
            raise SystemExit(f"Cache bulunamadi: {cache_path} -- once --from-cache OLMADAN calistirin")
        print(f"=== Cache'ten yukleniyor: {cache_path} (detection/tracking ATLANDI) ===")
        raw_rows = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        seq_dir = DATA_DIR / args.sequence
        gt_by_frame_full = _load_gt_all_visibility(seq_dir / "gt" / "gt.txt")
        seq_fps = read_seq_fps(seq_dir)

        print(f"=== {args.sequence} -- Faz 2.9/2.10 baseline hyp uretiliyor (conf={BASELINE_CONF}, track_thresh varsayilan) ===")
        hyp_by_frame_xyxy = _collect_baseline_hyp(seq_dir, seq_fps)

        events = compute_orr(gt_by_frame_full, hyp_by_frame_xyxy, MIN_OCCLUSION_FRAMES, MAX_RECOVERY_GAP, SEARCH_WINDOW)
        within = [e for e in events if e["within_window"]]
        print(f"  {len(within)} pencere-ici event bulundu")

        frames_needed = set()
        for e in within:
            frames_needed.add(e["start"] - 1)
            frames_needed.add(e["end"] + 1)
        print(f"\n=== {len(frames_needed)} benzersiz sinir karesinde ham tespit cikariliyor (conf={LOW_CONF}) ===")
        boundary_dets = _collect_boundary_detections(seq_dir, frames_needed)

        raw_rows = _collect_raw_features(within, gt_by_frame_full, hyp_by_frame_xyxy, boundary_dets)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(raw_rows, indent=2), encoding="utf-8")
        print(f"  Ham olculer cache'e yazildi: {cache_path} (siniflandirmayi degistirip --from-cache ile tekrar deneyin)")

    rows = _classify_raw_rows(raw_rows)

    print(f"\n{'gt_id':>5s} {'gap':>11s} {'len':>4s} {'vis(b/a)':>10s} {'area(b/a)':>14s} "
          f"{'crowd(b/a)':>10s} {'bestIoU(b/a)':>13s} {'bestConf(b/a)':>14s} {'reason(b/a)':>12s} {'LABEL':>10s}")
    for r in rows:
        print(f"{r['gt_id']:5d} {r['start']:5d}-{r['end']:<5d} {r['length']:4d} "
              f"{r['vis_before']:.2f}/{r['vis_after']:<.2f}   "
              f"{r['area_before']:6.0f}/{r['area_after']:<6.0f}   "
              f"{r['crowd_before']:3d}/{r['crowd_after']:<3d}      "
              f"{r['best_iou_before']:.2f}/{r['best_iou_after']:<.2f}        "
              f"{r['best_conf_before']:.2f}/{r['best_conf_after']:<.2f}         "
              f"{str(r['reason_before']):>3s}/{str(r['reason_after']):<3s}      "
              f"{r['label']:>10s}")

    print(f"\n=== Dagilim ({args.sequence}, event etiketi) ===")
    from collections import Counter
    counts = Counter(r["label"] for r in rows)
    for label in ("E", "A", "B", "C", "D", "RECOVERED", "UNKNOWN"):
        if counts.get(label):
            print(f"  {label:10s} {counts[label]}")

    vis_values = [r["vis_before"] for r in rows if r["label"] == "E"] + [r["vis_after"] for r in rows if r["label"] == "E"]
    if vis_values:
        print(f"  E etiketli taraflarin gorunurluk araligi: {min(vis_values):.2f}-{max(vis_values):.2f}")

    print(f"\n=== ORR ozeti ({args.sequence}, search_window={SEARCH_WINDOW}, ayni pencere-ici event'ler) ===")
    summarize_events(raw_rows, args.sequence)

    return rows


if __name__ == "__main__":
    main()
