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

`search_window` (varsayilan 5) -- before/after hyp_id ararken sinir karesinden
kac kare geriye/ileriye bakilacagi. ILK calistirmada (MOT17-04, search_window=5)
28 event'in HICBIRINDE gecerli eslesme bulunamadi; `evaluation/_debug_orr_diag.py`
ile arastirildi (script silindi, bulgular burada): global (kare,gt) eslesme
orani %34.7 (yani eslestirme mekanizmasi genelde CALISIYOR), ama tam sinir
karelerinde ozellikle basarisiz -- bazen o karede hic hyp kutusu yok (best_iou=0),
bazen 0.5 esigin altinda kismi bir kutu var (0.12-0.27), bazen esigi GECEN bir
aday var (0.677) ama ayni karedeki BASKA bir GT kimligi o hyp kutusunu daha
yuksek skorla "calmis" (kalabalik sahnede greedy eslestirmenin dogal sonucu).
Bu YANLIS DEGIL, gercek bir bulgu: yogun kalabalikta emerging-from-occlusion
anlarinda tracker'in KENDI cikisi birkac kare boyunca kararsiz. search_window'u
genisletmek bunu telafi eder ama pencereyi cok genisletmek "occlusion'dan
onceki/sonraki durum" yerine "iyice sonraki durum"u olcmeye baslar -- bu yuzden
birden fazla search_window degeriyle raporlaniyor (bkz. README "Faz 2.9").

Faz 2.10: `_match_gt_to_hyp` greedy'den Hungarian'a (`linear_sum_assignment`)
gecirildi -- Faz 2.9'daki 5 ornekten 1'i ("iyi aday baska kimlik tarafindan
kapildi") tam olarak bu fonksiyonun greedy davranisiydi; diger 4'u (hic
tespit yok / esik alti kismi ortusme) atama algoritmasiyla ilgisiz, detector
recall sorunu. Hungarian'a gecince MOT17-04'un olculebilirligi nasil
degisti/degismedi icin bkz. README "Faz 2.10" -- bu, "association failure"
hipotezini ELEMEK icin tasarlanmis bir test, iyilesme garantisi degil.

Kullanim:
    python -m evaluation.occlusion_recovery_rate --sequence MOT17-04-FRCNN
    python -m evaluation.occlusion_recovery_rate --sequence MOT17-04-FRCNN --checkpoint checkpoints/osnet_market1501/model/model.pth.tar-15
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from detection.yolo_detector import YoloDetector
from evaluation.mot_metrics import read_seq_fps
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
    """Global-optimal IoU eslestirme (Hungarian, `scipy.optimize.linear_sum_assignment`)
    -- {gt_id: hyp_id}. `evaluation/mot_metrics.py` (motmetrics) MOTA/IDF1 icin
    ayni prensiple eslesir -- bu fonksiyon artik onunla metodolojik olarak
    tutarli (Faz 2.10, bkz. README).

    Faz 2.9'da bu fonksiyonun ESKI (greedy, en yuksek skordan baslayan) hali,
    kalabalik bir karede gecerli (IoU>=0.5) bir adayin baska bir GT kimlik
    tarafindan -- o kimligin TOPLAMDA daha iyi bir secenegi olsa bile -- once
    kapilmasina yol acabiliyordu. Hungarian butun kombinasyonlari birlikte
    degerlendirip TOPLAM IoU'yu maksimize eden atamayi bulur, bu riski ortadan
    kaldirir. Bu, ORR olcumunu ETKILER ama tracker'in KENDI canli atamasini
    ETKILEMEZ -- `tracking/adaptive_tracker.py` ve `tracking/bytetrack_wrapper.py`
    zaten bu oturumdan once beri Hungarian kullaniyordu; degisen sadece bu
    DEGERLENDIRME betiginin GT<->hipotez eslestirmesi."""
    if not gt_rows or not hyp_rows:
        return {}
    gt_ids = [g[0] for g in gt_rows]
    hyp_ids = [h[0] for h in hyp_rows]
    cost = np.empty((len(gt_rows), len(hyp_rows)), dtype=float)
    for i, (_, gt_box, _vis) in enumerate(gt_rows):
        for j, (_, hyp_box) in enumerate(hyp_rows):
            cost[i, j] = 1.0 - iou_xyxy(gt_box, hyp_box)

    row_idx, col_idx = linear_sum_assignment(cost)
    result = {}
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= 1.0 - IOU_MATCH_THRESH:  # yani gercek IoU >= esik
            result[gt_ids[r]] = hyp_ids[c]
    return result


def compute_orr(gt_by_frame_full: dict, hyp_by_frame_xyxy: dict, min_occlusion_frames: int,
                 max_recovery_gap: int, search_window: int = SEARCH_WINDOW):
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
        for offset in range(0, search_window + 1):
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


def frame_level_match_rate(gt_by_frame_full: dict, hyp_by_frame_xyxy: dict) -> tuple:
    """Saglik kontrolu: gorunur (vis>=0.2) (kare,gt) ciftlerinin ne kadari
    O KAREDE herhangi bir hyp kutusuna IoU>=0.5 ile eslesiyor? Bu, ORR'un
    event-sinir istatistiklerini yorumlarken "eslestirme mekanizmasi genel
    olarak calisiyor mu" sorusunu ayri kontrol eder -- dusuk/sifir ORR'un
    "hicbir sey eslesmiyor" (kod hatasi) ile "sinir karelerinde ozellikle
    zor" (gercek bulgu) arasindaki farkini gostermek icin.
    Donus: (gorunur_cift_sayisi, eslesen_sayisi, oran)."""
    n_frames = max(gt_by_frame_full.keys()) if gt_by_frame_full else 0
    total_visible = 0
    total_matched = 0
    for frame in range(1, n_frames + 1):
        gt_rows = gt_by_frame_full.get(frame, [])
        hyp_rows = hyp_by_frame_xyxy.get(frame, [])
        matches = _match_gt_to_hyp(gt_rows, hyp_rows)
        for gt_id, _box, vis in gt_rows:
            if vis >= 0.2:
                total_visible += 1
                if gt_id in matches:
                    total_matched += 1
    rate = total_matched / total_visible if total_visible else float("nan")
    return total_visible, total_matched, rate


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
    parser.add_argument("--search-windows", default="5,15,30",
                         help="Virgulle ayrilmis, denenecek search_window degerleri (bkz. modul docstring)")
    args = parser.parse_args()
    search_windows = [int(v) for v in args.search_windows.split(",")]

    seq_dir = DATA_DIR / args.sequence
    gt_by_frame_full = _load_gt_all_visibility(seq_dir / "gt" / "gt.txt")
    seq_fps = read_seq_fps(seq_dir)  # Faz 2.8: fps-farkinda mesafe butcesi icin

    detector = YoloDetector(weights="yolov8n.pt", conf_threshold=0.35, classes=[0])
    checkpoint_path = str(args.checkpoint) if args.checkpoint.is_file() else None
    extractor = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=checkpoint_path)
    tracker = AdaptiveTracker(weight_mode="adaptive", base_weights=(0.4, 0.3, 0.3), fps=seq_fps)

    img_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    hyp_by_frame_xyxy = {}
    for i, img_path in enumerate(img_paths, start=1):
        frame = cv2.imread(str(img_path))
        detections = detector.detect(frame)
        person_dets = [d for d in detections if d.cls_name == "person"]
        tracks = tracker.update(person_dets, frame, extractor=extractor)
        hyp_by_frame_xyxy[i] = [(t.track_id, t.xyxy) for t in tracks]

    n_visible, n_matched, match_rate = frame_level_match_rate(gt_by_frame_full, hyp_by_frame_xyxy)
    print(f"\n=== {args.sequence} ({seq_fps:.0f} FPS) -- saglik kontrolu ===")
    print(f"  Genel kare-seviyesi eslesme orani: {n_matched}/{n_visible} ({match_rate:.1%}) "
          f"gorunur (kare,gt) cifti IoU>=0.5 ile eslesti (search_window'dan BAGIMSIZ, tum kareler)")

    for sw in search_windows:
        events = compute_orr(gt_by_frame_full, hyp_by_frame_xyxy, args.min_occlusion_frames,
                              args.max_recovery_gap, search_window=sw)
        print(f"\n=== {args.sequence} -- Occlusion Recovery Rate (min_occlusion_frames="
              f"{args.min_occlusion_frames}, max_recovery_gap={args.max_recovery_gap}, "
              f"search_window={sw}) ===")
        within = [e for e in events if e["within_window"]]
        beyond = [e for e in events if not e["within_window"]]
        summarize_events(within, "pencere ICINDE (kurtarma beklenir)")
        summarize_events(beyond, "pencere DISINDA (tasarim geregi kurtarilmasi beklenmez)")


if __name__ == "__main__":
    main()
