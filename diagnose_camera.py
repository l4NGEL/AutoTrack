"""Webcam icin calisan backend/fourcc kombinasyonunu bulan tanı scripti.

Kullanim:
    python diagnose_camera.py

Her kombinasyonu dener, bir kare yakalar, karenin "gurultu" olup olmadigini
(std sapma cok yuksekse gurultu) kontrol eder ve calisan kombinasyonlari
data/diag_<backend>_<fourcc>.png olarak kaydeder.
"""
import os

import cv2
import numpy as np

os.makedirs("data", exist_ok=True)

BACKENDS = [
    ("DEFAULT", 0),
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
]

FOURCCS = [None, "MJPG", "YUY2", "NV12", "I420"]


def looks_like_noise(frame) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # rastgele gurultu: komsu pikseller arasinda neredeyse hic korelasyon olmaz
    diff = np.abs(gray[:, 1:].astype(int) - gray[:, :-1].astype(int))
    return float(np.mean(diff)) > 40


for backend_name, backend_flag in BACKENDS:
    for fourcc_name in FOURCCS:
        cap = cv2.VideoCapture(0, backend_flag) if backend_flag else cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"[ACILAMADI] backend={backend_name} fourcc={fourcc_name}")
            cap.release()
            continue

        if fourcc_name:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_name))

        # kameranin oturmasi icin birkac kare at
        ok, frame = False, None
        for _ in range(5):
            ok, frame = cap.read()

        if not ok or frame is None:
            print(f"[KARE YOK]    backend={backend_name} fourcc={fourcc_name}")
            cap.release()
            continue

        noisy = looks_like_noise(frame)
        mean_val = float(np.mean(frame))
        status = "GURULTULU" if noisy else "TEMIZ"
        print(f"[{status:9s}] backend={backend_name:8s} fourcc={str(fourcc_name):5s} "
              f"shape={frame.shape} mean={mean_val:.1f}")

        if not noisy:
            out_path = f"data/diag_{backend_name}_{fourcc_name}.png"
            cv2.imwrite(out_path, frame)
            print(f"             -> kaydedildi: {out_path}")

        cap.release()

print("\nBitti. 'TEMIZ' olarak isaretlenen ve data/ altina kaydedilen PNG'leri kontrol et.")
