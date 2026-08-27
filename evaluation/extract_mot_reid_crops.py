"""MOT17_FINETUNE_SEQUENCES sekanslarindan, gt.txt'teki kimliklere gore
kirpilmis yaya goruntuleri cikarir -- OSNet'i MOT domaininde fine-tune
etmek icin zayif etiketli bir Re-ID veri kumesi olusturur.

Sadece download_datasets.MOT17_FINETUNE_SEQUENCES kullanilir (MOT17-09/11)
-- evaluation/run_ablation.py'de degerlendirilen MOT17-02/04'e KESINLIKLE
dokunulmaz (aksi halde fine-tune sonrasi ablation'da test-setine sizinti/
leakage olur, sonuclar sahte-iyi cikar).

Kullanim:
    python -m evaluation.extract_mot_reid_crops
"""
from pathlib import Path

import cv2

from evaluation.download_datasets import MOT17_FINETUNE_SEQUENCES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "mot17_reid_finetune"
MIN_VISIBILITY = 0.3  # egitim verisi icin dogruluk mot_metrics.py'nin 0.2'sinden biraz daha onemli
MIN_CROP_SIZE = 20  # cok kucuk kirpimlar Re-ID egitimi icin gurultu


def extract_sequence(seq_name: str) -> int:
    seq_dir = DATA_DIR / "MOT17" / seq_name
    gt_path = seq_dir / "gt" / "gt.txt"
    img_dir = seq_dir / "img1"
    if not gt_path.is_file():
        print(f"[extract] UYARI: {gt_path} yok, atlaniyor (once evaluation.download_datasets calistirin)")
        return 0

    by_frame = {}
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            x, y, w, h = (float(v) for v in parts[2:6])
            cls = int(float(parts[7])) if len(parts) > 7 else 1
            vis = float(parts[8]) if len(parts) > 8 else 1.0
            if cls != 1 or vis < MIN_VISIBILITY or w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
                continue
            by_frame.setdefault(frame, []).append((obj_id, x, y, w, h))

    n_saved = 0
    for frame, rows in by_frame.items():
        img_path = img_dir / f"{frame:06d}.jpg"
        if not img_path.is_file():
            continue
        frame_img = cv2.imread(str(img_path))
        if frame_img is None:
            continue
        fh, fw = frame_img.shape[:2]
        for obj_id, x, y, w, h in rows:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(fw, int(x + w)), min(fh, int(y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_img[y1:y2, x1:x2]
            identity_dir = OUT_DIR / f"{seq_name}_{obj_id:04d}"
            identity_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(identity_dir / f"{frame:06d}.jpg"), crop)
            n_saved += 1

    return n_saved


def main():
    total = 0
    for seq in MOT17_FINETUNE_SEQUENCES:
        n = extract_sequence(seq)
        print(f"[extract] {seq}: {n} kirpim")
        total += n

    identities = [d for d in OUT_DIR.iterdir() if d.is_dir()] if OUT_DIR.is_dir() else []
    print(f"[extract] toplam {total} kirpim, {len(identities)} benzersiz kimlik -> {OUT_DIR}")


if __name__ == "__main__":
    main()
