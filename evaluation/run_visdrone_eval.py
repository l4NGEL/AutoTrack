"""Faz 4.0: VisDrone-MOT sekanslarinda, GUNCEL en iyi (Faz 2.7-3.1) config
DEGISTIRILMEDEN, gercek MOTA/IDF1/ID-switch olcumu. `evaluation/
occlusion_event_forensics.py`'nin ayni baseline-hyp-uretme fonksiyonunu
(`_collect_baseline_hyp`) yeniden kullanir -- kod tekrari yok.

Kullanim:
    python -m evaluation.run_visdrone_eval --sequence uav0000305_00000_v
    python -m evaluation.run_visdrone_eval --sequence uav0000305_00000_v,uav0000137_00458_v
"""
import argparse
from pathlib import Path

from evaluation.mot_metrics import evaluate_sequence, load_mot_gt, read_seq_fps, summarize, xyxy_to_xywh
from evaluation.occlusion_event_forensics import BASELINE_CONF, _collect_baseline_hyp

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "VisDrone"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, help="Virgulle ayrilmis sekans adi/adlari")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    rows = []
    for seq in args.sequence.split(","):
        seq_dir = args.data_dir / seq
        gt_by_frame = load_mot_gt(seq_dir / "gt" / "gt.txt")
        seq_fps = read_seq_fps(seq_dir)
        print(f"\n=== {seq} ({seq_fps:.0f} FPS, conf={BASELINE_CONF}, track_thresh=0.5 varsayilan) ===")

        hyp_by_frame_xyxy = _collect_baseline_hyp(seq_dir, seq_fps)
        hyp_by_frame_xywh = {
            f: [(tid, xyxy_to_xywh(tuple(box))) for tid, box in fr_rows] for f, fr_rows in hyp_by_frame_xyxy.items()
        }
        acc = evaluate_sequence(gt_by_frame, hyp_by_frame_xywh)
        summary = summarize({seq: acc})
        r = summary.loc[seq]
        print(f"  MOTA={r['mota']:.3f} IDF1={r['idf1']:.3f} IDSW={int(r['num_switches'])} "
              f"FP={int(r['num_false_positives'])} FN={int(r['num_misses'])} GT_obj={int(r['num_objects'])}")
        rows.append({
            "sequence": seq, "mota": float(r["mota"]), "idf1": float(r["idf1"]),
            "idsw": int(r["num_switches"]), "fp": int(r["num_false_positives"]),
            "fn": int(r["num_misses"]), "gt_obj": int(r["num_objects"]),
        })

    return rows


if __name__ == "__main__":
    main()
