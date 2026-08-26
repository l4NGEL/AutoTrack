"""Kalman filtre tabanli sabit-hizli hareket modeli (izleme icin durum tahmini).

Durum vektoru: [cx, cy, w, h, vcx, vcy, vw, vh] (kutu merkezi, boyutu, hizlari)
Olcum vektoru: [cx, cy, w, h]
"""
from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter


def _xyxy_to_measurement(xyxy) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    return np.array([cx, cy, w, h], dtype=np.float64)


def _measurement_to_xyxy(cx, cy, w, h) -> tuple:
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class KalmanBoxTracker:
    """Tek bir hedef icin sabit hizli (constant velocity) Kalman filtresi.

    bytetrack_wrapper.ByteTracker'daki dogrudan IOU eslesmesinin aksine, bu
    filtre her karede bir sonraki konumu *tahmin eder* (predict) ve gercek
    tespitle guncellenir (update). Tahmin ile gercek tespit arasindaki fark
    (innovation), adaptive_tracker.py'de hem eslesme maliyetine hem de track
    guven skoruna (motion_conf) girdi olarak kullanilir.
    """

    def __init__(self, xyxy, dt: float = 1.0):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        self.kf.F = np.eye(8)
        for i in range(4):
            self.kf.F[i, i + 4] = dt

        self.kf.H = np.zeros((4, 8))
        self.kf.H[:4, :4] = np.eye(4)

        # Olcum gurultusu: merkez konumu boyuttan daha guvenilir olculur.
        self.kf.R = np.diag([1.0, 1.0, 10.0, 10.0])

        # Baslangic belirsizligi: hiz bilesenleri baslangicta cok belirsiz.
        self.kf.P *= 10.0
        self.kf.P[4:, 4:] *= 1000.0

        self.kf.Q = np.eye(8) * 0.01
        self.kf.Q[4:, 4:] *= 0.1

        self.kf.x[:4, 0] = _xyxy_to_measurement(xyxy)
        self.kf.x[4:, 0] = 0.0

        self.last_innovation_norm = 0.0
        self.age = 0

    def predict(self) -> tuple:
        self.kf.predict()
        self.age += 1
        w = max(float(self.kf.x[2, 0]), 1.0)
        h = max(float(self.kf.x[3, 0]), 1.0)
        cx, cy = float(self.kf.x[0, 0]), float(self.kf.x[1, 0])
        return _measurement_to_xyxy(cx, cy, w, h)

    def update(self, xyxy) -> None:
        z = _xyxy_to_measurement(xyxy)
        predicted = self.kf.x[:4, 0].copy()
        self.kf.update(z)
        # Olcum gurultusunun std sapmasina gore normalize edilmis innovation.
        std = np.sqrt(np.diag(self.kf.R))
        self.last_innovation_norm = float(np.linalg.norm((z - predicted) / std))

    @property
    def velocity(self) -> tuple:
        return (float(self.kf.x[4, 0]), float(self.kf.x[5, 0]))

    @property
    def velocity_magnitude(self) -> float:
        vx, vy = self.velocity
        return float(np.hypot(vx, vy))
