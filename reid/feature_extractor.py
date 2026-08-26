"""OSNet tabanli gorunum (appearance) embedding cikarici.

Kimlik kaybini onlemek icin her track'in gorunum imzasini cikarir; bu imza
tracking/adaptive_tracker.py'de Re-ID benzerlik skoru olarak kullanilir.
Sadece 'person' sinifi tespitler icin anlamlidir -- OSNet Market-1501/MSMT17
gibi insan Re-ID veri kumeleri uzerinde egitilir, arac/hayvan gibi baska
siniflara uygulanmasi gecersizdir (bkz. adaptive_tracker.REID_ELIGIBLE_CLASSES).
"""
from __future__ import annotations

import warnings

import cv2
import numpy as np
import torch
import torchreid


class OSNetFeatureExtractor:
    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        checkpoint_path: str | None = None,
        device: str | None = None,
        input_size: tuple = (256, 128),  # (yukseklik, genislik) -- Market-1501 standardi
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = tuple(input_size)

        pretrained = checkpoint_path is None
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,
            loss="softmax",
            pretrained=pretrained,
        )
        self.has_trained_weights = checkpoint_path is not None
        if checkpoint_path:
            torchreid.utils.load_pretrained_weights(self.model, checkpoint_path)
        else:
            warnings.warn(
                "OSNetFeatureExtractor: egitilmis Re-ID agirligi verilmedi, "
                "sadece ImageNet on-egitimli backbone kullaniliyor. Re-ID "
                "benzerlik skorlari bu haliyle zayif olacaktir -- "
                "checkpoint_path ile Market-1501/MSMT17 egitilmis agirlik saglayin.",
                stacklevel=2,
            )

        self.model.to(self.device).eval()

        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, *self.input_size, device=self.device)
            self._feat_dim = int(self.model(dummy).shape[1])

    @torch.no_grad()
    def extract(self, frame: np.ndarray, boxes_xyxy: list) -> np.ndarray:
        """Verilen kutular icin L2-normalize embedding matrisi (N, D) dondurur."""
        if not boxes_xyxy:
            return np.zeros((0, self._feat_dim), dtype=np.float32)

        batch = np.stack([self._preprocess(frame, box) for box in boxes_xyxy])
        tensor = torch.from_numpy(batch).to(self.device)
        features = self.model(tensor)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy()

    def _preprocess(self, frame: np.ndarray, box_xyxy: tuple) -> np.ndarray:
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x2 <= x1 or y2 <= y1:
            crop = np.zeros((*self.input_size, 3), dtype=np.uint8)
        else:
            crop = frame[y1:y2, x1:x2]

        crop = cv2.resize(crop, (self.input_size[1], self.input_size[0]))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - self._mean) / self._std
        return crop.transpose(2, 0, 1)  # HWC -> CHW


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)
