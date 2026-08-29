"""MOT17 tarzi gercek zemin dogrusuna karsi izleyici degerlendirmesi.

motmetrics (py-motmetrics) kutuphanesi ile CLEAR MOT metriklerini (MOTA,
ID switch sayisi, FP/FN) ve IDF1'i hesaplar.

HOTA bilinçli olarak KAPSAM DISINDA birakildi: resmi TrackEval
implementasyonu alpha-esigi uzerinden integral gerektiriyor ve bunu
ozensiz/dogrulanmamis bir sekilde yeniden yazip "HOTA" diye sunmak
yaniltici olur. MOTA/IDF1/ID-switch, motmetrics gibi bagimsiz ve yaygin
kullanilan bir kutuphaneden geldigi icin guvenilir olcumlerdir.

Degerlendirme MOT17 gt.txt'deki sadece class==1 (yaya) ve visibility>=0.2
satirlarini kullanir -- resmi MOTChallenge protokolunun distractor/occluder
sinif kurallarinin tamamini uygulamiyoruz (bilinçli, README'de belirtilen
bir basitlestirme).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

if not hasattr(np, "asfarray"):
    # motmetrics 1.4.0 (PyPI'deki en guncel surum, 2026-08) NumPy 2.0'da
    # kaldirilan np.asfarray'i cagiriyor (distances.py:117). numpy'i
    # dusurmek torch/ultralytics/opencv yiginini riske atar, bu yuzden
    # geriye-donuk uyumluluk shim'i tercih edildi.
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm  # noqa: E402  (shim'den sonra import edilmeli)

PEDESTRIAN_CLASS = 1
MIN_VISIBILITY = 0.2


def read_seq_fps(seq_dir, default: float = 30.0) -> float:
    """seqinfo.ini'deki frameRate'i okur (Faz 2.7: AdaptiveTracker'in
    fps-farkinda mesafe olceklemesi icin gercek kare hizi lazim -- bkz.
    tracking/adaptive_tracker.py). Dosya/alan yoksa default'a duser."""
    seqinfo = Path(seq_dir) / "seqinfo.ini"
    if not seqinfo.is_file():
        return default
    for line in seqinfo.read_text(encoding="utf-8").splitlines():
        if line.strip().lower().startswith("framerate"):
            try:
                return float(line.split("=", 1)[1].strip())
            except (IndexError, ValueError):
                return default
    return default


def load_mot_gt(gt_path) -> dict:
    """MOT formatli gt.txt okur -> {frame: [(obj_id, (x,y,w,h)), ...]}."""
    by_frame: dict = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            x, y, w, h = (float(v) for v in parts[2:6])
            cls = int(float(parts[7])) if len(parts) > 7 else PEDESTRIAN_CLASS
            vis = float(parts[8]) if len(parts) > 8 else 1.0
            if cls != PEDESTRIAN_CLASS or vis < MIN_VISIBILITY:
                continue
            by_frame.setdefault(frame, []).append((obj_id, (x, y, w, h)))
    return by_frame


def xyxy_to_xywh(xyxy) -> tuple:
    x1, y1, x2, y2 = xyxy
    return (x1, y1, x2 - x1, y2 - y1)


def evaluate_sequence(gt_by_frame: dict, hyp_by_frame: dict) -> mm.MOTAccumulator:
    """hyp_by_frame: {frame: [(track_id, (x,y,w,h)), ...]} -- xywh formatinda."""
    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt_by_frame) | set(hyp_by_frame))
    for frame in all_frames:
        gt = gt_by_frame.get(frame, [])
        hyp = hyp_by_frame.get(frame, [])
        gt_ids = [g[0] for g in gt]
        hyp_ids = [h[0] for h in hyp]
        gt_boxes = np.array([g[1] for g in gt], dtype=float) if gt else np.zeros((0, 4))
        hyp_boxes = np.array([h[1] for h in hyp], dtype=float) if hyp else np.zeros((0, 4))
        dists = mm.distances.iou_matrix(gt_boxes, hyp_boxes, max_iou=0.5)
        acc.update(gt_ids, hyp_ids, dists)
    return acc


METRIC_NAMES = ["mota", "idf1", "num_switches", "num_false_positives", "num_misses", "num_objects"]


def summarize(accumulators: dict) -> "mm.pandas.DataFrame":
    mh = mm.metrics.create()
    return mh.compute_many(
        list(accumulators.values()),
        metrics=METRIC_NAMES,
        names=list(accumulators.keys()),
        generate_overall=len(accumulators) > 1,
    )


def format_summary(summary) -> str:
    strsummary = mm.io.render_summary(
        summary,
        formatters=mm.metrics.create().formatters,
        namemap={
            "mota": "MOTA",
            "idf1": "IDF1",
            "num_switches": "ID Sw.",
            "num_false_positives": "FP",
            "num_misses": "FN (miss)",
            "num_objects": "GT obj.",
        },
    )
    return strsummary
