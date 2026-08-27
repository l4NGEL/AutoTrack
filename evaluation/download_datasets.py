"""Faz 2 degerlendirmesi icin gerekli verileri indirir.

motchallenge.net ve torchreid'in varsayilan Market-1501 mirror'i
(188.138.127.15) bu gelistirme ortamindan erisilemedi (agla ilgili
red/timeout) -- onlarin yerine ayni verinin dogrulanmis HuggingFace Hub
kopyalarini kullaniyoruz:

  - Market-1501: https://huggingface.co/datasets/Lekim89/market1501
    (resmi "Market-1501-v15.09.15.zip" ile ayni ad/boyut araliginda tar.gz)
  - MOT17 (train, FRCNN tespit varyanti): https://huggingface.co/datasets/Morrison1025/MOT17
    (gt.txt + det.txt + img1/ iceren standart MOT dizin yapisi, 7 sekansin
    tamami mevcut; biz sadece MOT17-02 ve MOT17-04'u indiriyoruz)

Kullanim:
    python -m evaluation.download_datasets
"""
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MOT17_EVAL_SEQUENCES = ["MOT17-02-FRCNN", "MOT17-04-FRCNN"]  # evaluation/run_ablation.py -- asla egitimde kullanilmaz
MOT17_FINETUNE_SEQUENCES = ["MOT17-09-FRCNN", "MOT17-11-FRCNN"]  # evaluation/finetune_reid_mot.py -- asla degerlendirmede kullanilmaz


def download_market1501():
    target = DATA_DIR / "market1501"
    marker = target / "Market-1501-v15.09.15" / "bounding_box_train"
    if marker.is_dir():
        print(f"[market1501] zaten mevcut: {marker}")
        return

    print("[market1501] Lekim89/market1501'den indiriliyor (~110MB)...")
    archive_path = hf_hub_download(
        repo_id="Lekim89/market1501",
        repo_type="dataset",
        filename="Market-1501-v15.09.15.tar.gz",
        local_dir=str(target),
    )

    print(f"[market1501] cikartiliyor: {archive_path}")
    with tarfile.open(archive_path) as tar:
        tar.extractall(target)

    if marker.is_dir():
        print(f"[market1501] tamam: {marker}")
    else:
        found = list(target.rglob("bounding_box_train"))
        print(f"[market1501] UYARI: beklenen yol yok, bulunanlar: {found}")


def download_mot17(sequences=MOT17_EVAL_SEQUENCES):
    target = DATA_DIR / "MOT17"
    for seq in sequences:
        marker = target / seq / "gt" / "gt.txt"
        if marker.is_file():
            print(f"[MOT17] {seq} zaten mevcut")
            continue

        print(f"[MOT17] {seq} indiriliyor (Morrison1025/MOT17'den)...")
        snapshot_download(
            repo_id="Morrison1025/MOT17",
            repo_type="dataset",
            allow_patterns=[f"ablation/{seq}/*"],
            local_dir=str(target / "_raw"),
        )

    # HF deposundaki "ablation/<seq>/..." onekini duz "<seq>/..." yapisina tasi.
    raw_root = target / "_raw" / "ablation"
    if raw_root.is_dir():
        for seq_dir in raw_root.iterdir():
            dest = target / seq_dir.name
            if not dest.exists():
                seq_dir.rename(dest)
        remaining = list((target / "_raw").rglob("*"))
        if not remaining or all(p.is_dir() for p in remaining):
            import shutil

            shutil.rmtree(target / "_raw", ignore_errors=True)

    for seq in sequences:
        gt = target / seq / "gt" / "gt.txt"
        n_frames = len(list((target / seq / "img1").glob("*.jpg"))) if (target / seq / "img1").is_dir() else 0
        print(f"[MOT17] {seq}: gt.txt {'var' if gt.is_file() else 'YOK'}, {n_frames} kare")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_market1501()
    download_mot17(MOT17_EVAL_SEQUENCES)
    download_mot17(MOT17_FINETUNE_SEQUENCES)
