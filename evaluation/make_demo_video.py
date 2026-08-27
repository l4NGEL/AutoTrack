"""MOT17 sekans karelerini tek bir demo video dosyasina birlestirir.

Canli webcam yerine gercek, kalabalik bir sokak sahnesi kullanmak icin --
boylece AutoTrack'i calistirinca gercekten insan/takip/ID gormek garanti,
bos bir oda degil.

Kullanim:
    python -m evaluation.make_demo_video                  # MOT17-04-FRCNN (varsayilan)
    python -m evaluation.make_demo_video --sequence MOT17-02-FRCNN
"""
import argparse
from pathlib import Path

import cv2

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_SEQUENCE = "MOT17-04-FRCNN"
FPS = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE, help="MOT17/<sequence>/img1 klasoru")
    args = parser.parse_args()

    seq_dir = DATA_DIR / "MOT17" / args.sequence
    out_path = DATA_DIR / f"demo_{args.sequence.lower().replace('-', '_')}.mp4"

    img_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    if not img_paths:
        raise SystemExit(f"Kare bulunamadi: {seq_dir / 'img1'} -- once evaluation.download_datasets calistirin.")

    first = cv2.imread(str(img_paths[0]))
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (w, h))

    for p in img_paths:
        frame = cv2.imread(str(p))
        writer.write(frame)
    writer.release()

    print(f"[make_demo_video] {len(img_paths)} kare -> {out_path} ({w}x{h}, {FPS} FPS, "
          f"~{len(img_paths)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
