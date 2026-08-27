"""OSNet'i Market-1501 uzerinde gercekten egitir (ImageNet backbone'dan
transfer learning) ve rank-1/mAP raporlar.

Bu, torchreid'in kendi yuksek seviyeli egitim motorunu (ImageSoftmaxEngine)
kullanir -- elle yazilmis bir egitim dongusu degil, kutuphanenin standart,
test edilmis yolu.

Kullanim:
    python -m evaluation.train_reid --epochs 15
"""
import argparse
from pathlib import Path

import torchreid

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "osnet_market1501"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0015)
    args = parser.parse_args()

    datamanager = torchreid.data.ImageDataManager(
        root=str(DATA_DIR),
        sources="market1501",
        height=256,
        width=128,
        batch_size_train=args.batch_size,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"],
    )

    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True,  # ImageNet backbone'dan basla (transfer learning)
    )
    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(model, optim="adam", lr=args.lr)
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler="single_step", stepsize=max(args.epochs - 5, 1)
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager, model, optimizer=optimizer, scheduler=scheduler, label_smooth=True
    )

    print(f"[train_reid] {args.epochs} epoch, batch={args.batch_size}, lr={args.lr}, "
          f"train_pid={datamanager.num_train_pids}")

    engine.run(
        save_dir=str(CHECKPOINT_DIR),
        max_epoch=args.epochs,
        eval_freq=args.epochs,  # sadece sonda degerlendir (zamandan tasarruf)
        print_freq=50,
        test_only=False,
    )

    print(f"[train_reid] bitti. Checkpoint: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
