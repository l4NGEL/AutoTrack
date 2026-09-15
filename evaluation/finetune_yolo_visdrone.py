"""Faz 4.1: yolov8n.pt'yi VisDrone'un KENDI egitim sahneleriyle (Faz 4.0'in
degerlendirme sahnelerine HIC DOKUNMADAN, bkz. prepare_yolo_finetune_data.py)
fine-tune eder -- Faz 4.0'in bulgusuna dogrudan cevap: COCO-egitimli
agirliklar hava-acisi/kucuk-olcekli (~20-30px) insanlari pratikte hic
tespit edemiyordu.

Tasarim kararlari:
  - COCO agirliklarindan DEVAM (sifirdan degil) -- diger 79 sinifi tamamen
    unutturmamak icin dusuk ogrenme orani (lr0 kucuk) ve mutevazi epoch
    sayisi (Faz 2.3'un ReID fine-tune'uyla ayni felsefe: "sifirdan degil,
    ustune ekleme").
  - imgsz=1280 (Ultralytics varsayilani 640 DEGIL): Faz 4.0'in kok nedeni
    tam olarak "insanlar cok kucuk" oldugu icin, 640'a kucultmek onlari
    egitim sirasinda da neredeyse gorunmez kilardi -- buyuk imgsz, az
    piksel detayini korumanin en dogrudan yolu.
  - Egitim verisi SADECE 'person' sinifi iceriyor (VisDrone-MOT sadece
    yaya/insan etiketliyor) -- diger 79 COCO sinifi icin egitim ornegi YOK,
    yani bu fine-tune'un diger siniflardaki performansi TEST EDILMIYOR
    (dogrustce not edilmesi gereken bir sinir, README'ye tasinacak).

Kullanim:
    python -m evaluation.finetune_yolo_visdrone --epochs 30
"""
import argparse
import os
from pathlib import Path

# Bu makinede Ultralytics'in kendi MLflow auto-log callback'i ACIK
# (settings['mlflow']=True -- baska bir projede, orn. Baykar NLP RAG/QLoRA/
# DPO calismasinda kullaniliyor olabilir) ve kurulu mlflow surumu artik
# dosya-sistemi (./mlruns) tabanli izlemeyi REDDEDIYOR. Bunu GLOBAL olarak
# (settings.update) degil, SADECE bu surec icin env degiskeniyle asiyoruz --
# ultralytics'in kendi hata mesajinin onerdigi tam kacis yolu, ve diger
# projelerin MLflow kurulumunu ETKILEMEZ.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from ultralytics import YOLO

DATA_YAML = Path(__file__).resolve().parent.parent / "data" / "visdrone_yolo_finetune" / "data.yaml"
PROJECT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "yolo_visdrone_finetune"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.0005, help="Faz 2.3'un ReID fine-tune'undan dusuk (0.0015->0.0003 mantigi) ile ayni ruhta -- sifirdan egitimin (COCO'nun kendi lr0'i ~0.01) cok altinda")
    parser.add_argument("--data-yaml", type=Path, default=DATA_YAML)
    args = parser.parse_args()

    if not args.data_yaml.is_file():
        raise SystemExit(f"Once 'python -m evaluation.prepare_yolo_finetune_data' calistirin -- {args.data_yaml} yok")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(args.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        project=str(PROJECT_DIR.parent),
        name=PROJECT_DIR.name,
        exist_ok=True,
        patience=10,  # 10 epoch iyilesme olmazsa erken dur -- kucuk veri kumesinde asiri ezberlemeyi sinirlamak icin
        val=True,
    )
    best_path = PROJECT_DIR / "weights" / "best.pt"
    print(f"\n=== Fine-tune tamamlandi ===")
    print(f"  En iyi checkpoint: {best_path}")
    return results


if __name__ == "__main__":
    main()
