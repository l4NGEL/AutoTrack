"""OSNet'i (Faz 2'nin Market-1501 checkpoint'inden baslayarak) MOT17'nin
kendi kimlikleriyle (evaluation/extract_mot_reid_crops.py ciktisi)
domain-adaptive fine-tune eder.

Faz 2 ablation'inin (MOT17-02/04) ana bulgusu domain gap'ti: OSNet
Market-1501'in kirpilmis/kontrollu yaya goruntuleri uzerinde egitildi,
MOT17 tamamen farkli bir domain (guvenlik kamerasi acisi, cozunurluk,
hareket bulanikligi). Bu script backbone'u Market-1501'den devralir,
sadece MOT17-09/11'in (degerlendirmede kullanilmayan, bkz.
extract_mot_reid_crops.py) kimlikleriyle ince ayar yapar.

Kullanim:
    python -m evaluation.extract_mot_reid_crops
    python -m evaluation.finetune_reid_mot --epochs 15
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchreid
from torch.utils.data import DataLoader, Dataset

import reid.feature_extractor  # noqa: F401  (yan etki: torchreid'in load_checkpoint'ini torch2.6 icin yamalar)

CROPS_DIR = Path(__file__).resolve().parent.parent / "data" / "mot17_reid_finetune"
SRC_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "osnet_market1501" / "model" / "model.pth.tar-15"
OUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "osnet_mot_finetuned"

INPUT_SIZE = (256, 128)  # (yukseklik, genislik)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MotIdentityDataset(Dataset):
    def __init__(self, crops_dir: Path, augment: bool = True):
        self.identities = sorted(d.name for d in crops_dir.iterdir() if d.is_dir())
        self.id_to_label = {name: i for i, name in enumerate(self.identities)}
        self.samples = [
            (img_path, self.id_to_label[identity])
            for identity in self.identities
            for img_path in (crops_dir / identity).glob("*.jpg")
        ]
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (INPUT_SIZE[1], INPUT_SIZE[0]))
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img.copy()), label

    @property
    def num_classes(self) -> int:
        return len(self.identities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)  # Faz 2'nin 0.0015'inden dusuk -- sifirdan degil, ince ayar
    args = parser.parse_args()

    if not CROPS_DIR.is_dir() or not any(CROPS_DIR.iterdir()):
        raise SystemExit(f"Kirpim bulunamadi: {CROPS_DIR} -- once 'python -m evaluation.extract_mot_reid_crops' calistirin.")

    dataset = MotIdentityDataset(CROPS_DIR)
    print(f"[finetune_reid_mot] {len(dataset)} kirpim, {dataset.num_classes} kimlik")
    if dataset.num_classes < 2:
        raise SystemExit("En az 2 kimlik gerekli (cross-entropy siniflandirma icin).")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=dataset.num_classes, loss="softmax", pretrained=False
    )
    torchreid.utils.load_pretrained_weights(model, str(SRC_CHECKPOINT))  # Market1501 backbone'u devral
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs - 5, 1), gamma=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    for epoch in range(args.epochs):
        total_loss, total_correct, total_n = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += imgs.size(0)
        scheduler.step()
        acc = 100.0 * total_correct / total_n
        print(f"[finetune_reid_mot] epoch {epoch + 1}/{args.epochs}  loss={total_loss / total_n:.4f}  acc={acc:.1f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "model.pth.tar"
    torch.save({"state_dict": model.state_dict(), "epoch": args.epochs}, out_path)
    print(f"[finetune_reid_mot] checkpoint kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
