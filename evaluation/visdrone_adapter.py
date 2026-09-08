"""Faz 4.0: VisDrone-MOT'u (Hugging Face `Voxel51/visdrone-mot` mirror'i,
`samples.json`) MOT17 ile AYNI dizin/format yapisina (`img1/*.jpg` +
`gt/gt.txt` + `seqinfo.ini`) cevirir -- boylece TUM mevcut `evaluation/*.py`
scriptleri (`mot_metrics.py`, `occlusion_recovery_rate.py`,
`occlusion_event_forensics.py`, `partial_visibility_forensics.py`,
`run_confidence_sweep.py`) HİÇ DEĞİŞTİRİLMEDEN VisDrone'da da calisir.

**Durustce belirtilmesi gereken yaklasikliklar (README'de de tekrarlanacak):**

1. `visibility` alani VisDrone'da (bu mirror'da) İKİLİ (0 veya 1), MOT17'nin
   0.05-1.0 SUREKLI degeri GİBİ DEĞİL. Dogrudan 0.0/1.0 olarak MOT gt.txt'e
   tasiniyor -- bu, Faz 2.9-3.1'in agir-occlusion (`vis<0.2`) analizini
   DOGRU calistirir (0 < 0.2 -> occluded, 1 >= 0.2 -> visible) ama Faz 3.3'un
   kismi-gorunurluk (`0.2<=vis<0.5`) analizini YAPISAL OLARAK İMKANSIZ
   kilar -- o araliga hicbir deger dusemez (ne bug ne veri hatasi, veri
   kumesinin kendi semasinin sinirini).
2. `frameRate` icin bu HF mirror'inda VisDrone'un ORİJİNAL cekim hizina
   dair dogrudan bir alan YOK -- `metadata.json`'daki FiftyOne app-config
   varsayilani `dynamic_groups_target_frame_rate=30` kullaniliyor. Bu bir
   YAKLASIKLIK (FiftyOne'in kendi UI-oynatma varsayilani), VisDrone'un
   gercek cekim hizinin dogrulanmis degeri DEGIL.
3. Sadece `pedestrian` ve `people` kategorileri tutuluyor (kullanicinin
   "vasita degil" tercihiyle tutarli); `ignored_region` ve tum vasita
   siniflari (car/van/truck/bus/motor/bicycle/tricycle/awning-tricycle)
   ATILIYOR.

Kullanim:
    python -m evaluation.visdrone_adapter --scenes uav0000305_00000_v,uav0000137_00458_v
"""
import argparse
import json
import shutil
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "visdrone_raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "VisDrone"
PERSON_LABELS = {"pedestrian", "people"}
APPROX_FRAME_RATE = 30  # bkz. modul docstring madde 2 -- dogrulanmis VisDrone cekim hizi DEGIL


def visdrone_bbox_to_pixel_xywh(bounding_box, img_width: float, img_height: float) -> tuple:
    """FiftyOne normalize [x,y,w,h] (0-1 araligi) -> piksel (x,y,w,h)."""
    nx, ny, nw, nh = bounding_box
    return nx * img_width, ny * img_height, nw * img_width, nh * img_height


def detection_to_gt_row(frame_number: int, detection: dict, img_width: float, img_height: float) -> str | None:
    """Bir VisDrone detection'ini MOT-tarzi gt.txt satirina cevirir
    (`frame,id,x,y,w,h,conf,cls,vis`), ya da person-disi/ignored_region
    ise None doner (cagiran taraf bu satiri atlamali)."""
    label = detection.get("label")
    if label not in PERSON_LABELS:
        return None
    x, y, w, h = visdrone_bbox_to_pixel_xywh(detection["bounding_box"], img_width, img_height)
    gt_id = detection["index"]
    vis = float(detection.get("visibility", 1))  # IKILI (0/1) -- bkz. modul docstring madde 1
    # conf=1 (gercek GT satiri), cls=1 (MOT'un "pedestrian" kodu -- occlusion_recovery_rate.py'nin filtresiyle uyumlu)
    return f"{frame_number},{gt_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,{vis:.2f}"


def build_seqinfo(name: str, img_width: int, img_height: int, seq_length: int) -> str:
    return (
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate={APPROX_FRAME_RATE}\n"
        f"seqLength={seq_length}\nimWidth={img_width}\nimHeight={img_height}\nimExt=.jpg\n"
    )


def build_sequence(scene_id: str, samples: list, out_dir: Path = OUT_DIR, raw_dir: Path = RAW_DIR) -> Path:
    """Bir VisDrone sahnesini MOT17-tarzi bir dizine (img1/gt/seqinfo.ini) yazar."""
    rows = [s for s in samples if s["scene_id"] == scene_id]
    rows.sort(key=lambda s: s["frame_number"])
    if not rows:
        raise ValueError(f"scene_id bulunamadi: {scene_id}")

    seq_dir = out_dir / scene_id
    (seq_dir / "img1").mkdir(parents=True, exist_ok=True)
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)

    gt_lines = []
    img_w = img_h = None
    for s in rows:
        meta = s.get("metadata") or {}
        img_w, img_h = meta.get("width", img_w), meta.get("height", img_h)
        frame_number = s["frame_number"]
        for det in s.get("detections", []):
            line = detection_to_gt_row(frame_number, det, img_w, img_h)
            if line is not None:
                gt_lines.append(line)

        src = raw_dir / s["filepath"]
        dst = seq_dir / "img1" / f"{frame_number:06d}.jpg"
        if src.is_file() and not dst.is_file():
            shutil.copyfile(src, dst)

    (seq_dir / "gt" / "gt.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    (seq_dir / "seqinfo.ini").write_text(
        build_seqinfo(scene_id, int(img_w), int(img_h), len(rows)), encoding="utf-8"
    )
    return seq_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", required=True, help="Virgulle ayrilmis scene_id listesi")
    parser.add_argument("--samples-json", type=Path, default=RAW_DIR / "samples.json")
    args = parser.parse_args()

    with open(args.samples_json, encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]

    for scene_id in args.scenes.split(","):
        seq_dir = build_sequence(scene_id, samples)
        n_frames = len(list((seq_dir / "img1").glob("*.jpg")))
        n_gt_lines = sum(1 for _ in open(seq_dir / "gt" / "gt.txt", encoding="utf-8")) - 1
        print(f"{scene_id}: {n_frames} kare, {n_gt_lines} gt satiri -> {seq_dir}")


if __name__ == "__main__":
    main()
