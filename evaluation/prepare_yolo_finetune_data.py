"""Faz 4.1: VisDrone-MOT egitim sahnelerini (zaten visdrone_adapter.py ile
MOT17-tarzi img1/+gt.txt formatina cevrilmis) Ultralytics YOLO tespit
egitim formatina (images/+labels/+data.yaml) cevirir.

**Sizinti (leakage) onlemi:** Faz 4.0'in degerlendirme sahneleri
(uav0000137_00458_v, uav0000305_00000_v) burada KULLANILMIYOR -- sadece
kullanicinin onayladigi 2 YENI sahne (uav0000086_00000_v [high density],
uav0000339_00001_v [low density]), Faz 2.3'un MOT17-09/11-vs-02/04 ayrimiyla
AYNI disiplinle.

**Train/val bolunmesi -- rastgele DEGIL:** her sahnenin kendi karelerinin
ilk %85'i train, son %15'i val. Rastgele (frame-bazli) bir bolunme, birbirine
COK yakin/neredeyse ozdes ardisik kareleri train VE val'e dagitip val
metrigini yapay sekilde sisirirdi -- tek bir zaman noktasindan bolmek bu
sizintiyi buyuk olcude azaltir (video ic-sekans korelasyonu, ReID
literaturunde bilinen bir sizinti turu). Bu val seti egitim sirasinda
"asiri ezberleme" kontrolu icindir -- ASIL genelleme testi hala Faz 4.0'in
DOKUNULMAMIS 2 sahnesi (bu script'in hic gormedigi).

Kullanim:
    python -m evaluation.prepare_yolo_finetune_data
"""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "VisDrone"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "visdrone_yolo_finetune"
TRAIN_SCENES = ["uav0000086_00000_v", "uav0000339_00001_v"]
VAL_FRACTION = 0.15


def _read_seqinfo_size(seq_dir: Path) -> tuple[int, int]:
    text = (seq_dir / "seqinfo.ini").read_text(encoding="utf-8")
    kv = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    return int(kv["imWidth"]), int(kv["imHeight"])


def _read_gt_by_frame(seq_dir: Path) -> dict:
    gt_by_frame: dict = {}
    for line in (seq_dir / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        frame, _id, x, y, w, h, _conf, _cls, _vis = line.split(",")
        gt_by_frame.setdefault(int(frame), []).append((float(x), float(y), float(w), float(h)))
    return gt_by_frame


def _write_yolo_label(dst_txt: Path, boxes_px: list, img_w: int, img_h: int) -> None:
    lines = []
    for x, y, w, h in boxes_px:
        cx, cy = (x + w / 2) / img_w, (y + h / 2) / img_h
        nw, nh = w / img_w, h / img_h
        # gecersiz/kirpilan kutulari at (goruntu disina tasan negatif/0 boyut)
        if nw <= 0 or nh <= 0:
            continue
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        nw, nh = min(nw, 1.0), min(nh, 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    dst_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build(train_scenes: list = TRAIN_SCENES, out_dir: Path = OUT_DIR, data_dir: Path = DATA_DIR) -> dict:
    counts = {"train": 0, "val": 0, "train_boxes": 0, "val_boxes": 0}
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for scene in train_scenes:
        seq_dir = data_dir / scene
        img_w, img_h = _read_seqinfo_size(seq_dir)
        gt_by_frame = _read_gt_by_frame(seq_dir)
        img_paths = sorted((seq_dir / "img1").glob("*.jpg"))
        n_val = max(1, int(len(img_paths) * VAL_FRACTION))
        split_point = len(img_paths) - n_val

        for idx, img_path in enumerate(img_paths):
            split = "train" if idx < split_point else "val"
            frame_number = int(img_path.stem)
            boxes = gt_by_frame.get(frame_number, [])

            stem = f"{scene}_{frame_number:06d}"
            dst_img = out_dir / "images" / split / f"{stem}.jpg"
            dst_txt = out_dir / "labels" / split / f"{stem}.txt"
            if not dst_img.is_file():
                dst_img.write_bytes(img_path.read_bytes())
            _write_yolo_label(dst_txt, boxes, img_w, img_h)

            counts[split] += 1
            counts[f"{split}_boxes"] += len(boxes)

    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: person\n",
        encoding="utf-8",
    )
    return counts


if __name__ == "__main__":
    counts = build()
    print(f"train: {counts['train']} kare, {counts['train_boxes']} kutu")
    print(f"val:   {counts['val']} kare, {counts['val_boxes']} kutu")
    print(f"-> {OUT_DIR}")
