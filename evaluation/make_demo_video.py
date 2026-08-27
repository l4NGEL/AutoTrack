"""MOT17-04-FRCNN karelerini tek bir demo video dosyasina birlestirir.

Canli webcam yerine gercek, kalabalik bir sokak sahnesi kullanmak icin --
boylece AutoTrack'i calistirinca gercekten insan/takip/ID gormek garanti,
bos bir oda degil.

Kullanim:
    python -m evaluation.make_demo_video
"""
from pathlib import Path

import cv2

SEQ_DIR = Path(__file__).resolve().parent.parent / "data" / "MOT17" / "MOT17-04-FRCNN"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_mot17_04.mp4"
FPS = 30


def main():
    img_paths = sorted((SEQ_DIR / "img1").glob("*.jpg"))
    if not img_paths:
        raise SystemExit(f"Kare bulunamadi: {SEQ_DIR / 'img1'} -- once evaluation.download_datasets calistirin.")

    first = cv2.imread(str(img_paths[0]))
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (w, h))

    for p in img_paths:
        frame = cv2.imread(str(p))
        writer.write(frame)
    writer.release()

    print(f"[make_demo_video] {len(img_paths)} kare -> {OUT_PATH} ({w}x{h}, {FPS} FPS, "
          f"~{len(img_paths)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
