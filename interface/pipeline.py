"""interface/app.py icin, main.py'nin CLI akisi disinda tekrar kullanilabilir
tespit + takip + oncelik pipeline sarmalayicisi.

main.py'ye HICBIR DOKUNUS yok -- ayni public siniflari (YoloDetector,
AdaptiveTracker, OSNetFeatureExtractor, PriorityEngine) main.py'nin
kullandigi sirayla, bagimsiz olarak cagirir. Amac tek: main.py'nin
dogrulanmis calisma mantigini KOPYALAMADAN, video dosyasi / PC kamerasi /
IP kamerasi gibi farkli kaynaklarin hepsinin ayni "kare -> overlay'li kare"
fonksiyonunu paylasmasi.

Hassas bolge (zone) kasitli olarak devre disi: config.yaml/config_demo.yaml
icindeki poligonlar belirli bir sahnenin cozunurlugune gore elle ayarlanmis
(orn. 1920x1080 icin [700,300]-[1400,900]) -- rastgele yuklenen bir video
veya webcam goruntusune uygulanirsa anlamsiz/yanlis yerde bir dikdortgen
cizer. Hiz + bekleme suresi bilesenleri hala calisir, sadece bolge bileseni
bu arayuzde her zaman 0'dir.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Bu dosya `interface/app.py` tarafindan "python interface/app.py" ile
# calistirildiginda bagimsiz bir kok modul olarak import edilir (sys.path[0]
# o zaman interface/ olur, AutoTrack/ degil) -- ama asagidaki satirlar
# (detection.*, tracking.*, ...) AutoTrack/ kokunun sys.path'te olmasini
# gerektirir. Proje kokunu garantiye almak icin ekliyoruz; main.py'yi
# calistiran normal "python main.py" akisini ETKILEMEZ.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from decision.priority_engine import PriorityEngine
from detection.yolo_detector import YoloDetector
from reid.feature_extractor import OSNetFeatureExtractor
from tracking.adaptive_tracker import AdaptiveTracker
from viz.overlay_renderer import draw_hud, draw_tracks

# Mutlak yol -- main.py'nin `yolov8n.pt` gibi cwd-bagimli goreli yolundan
# BILEREK farkli: main.py'nin belgelenen kullanimi zaten AutoTrack/ kokunden
# "python main.py" ile calistirmak, ama arayuz bir IDE/arac tarafindan farkli
# bir calisma dizininden baslatilabilir -- o durumda goreli "yolov8n.pt"
# ultralytics'in onu sessizce GitHub'dan yeniden indirmesine yol acar.
DEFAULT_WEIGHTS = str(_PROJECT_ROOT / "yolov8n.pt")
# Faz 2.3: MOT-domaininde fine-tune edilmis checkpoint, Faz 2'nin ham
# Market-1501 checkpoint'inden daha iyi -- config_demo.yaml ile ayni secim.
DEFAULT_REID_CHECKPOINT = str(_PROJECT_ROOT / "checkpoints" / "osnet_mot_finetuned" / "model.pth.tar")
PRIORITY_WEIGHTS = {"speed": 0.4, "zone": 0.4, "dwell_time": 0.2}

# YOLO/OSNet agirliklarini diskten yuklemek birkac saniye surebiliyor --
# arayuzde "video yukle" / "webcam baslat" / "IP kameraya baglan" HER
# tikta yeniden yuklemek yerine, ayni (agirlik, sinif-filtresi) kombinasyonu
# icin surec boyunca BIR KEZ yukleyip paylasiyoruz. AdaptiveTracker (track
# ID sayaci, kayip galeri) bu onbellege GIRMEZ -- o her yeni kaynak icin
# TrackingPipeline.__init__ tarafinda taze olusturulur.
_detector_cache: dict[tuple, YoloDetector] = {}
_reid_cache: dict[str | None, tuple[OSNetFeatureExtractor, bool]] = {}


class TrackingPipeline:
    """Tek bir video kaynagi icin durum tutan (stateful) sarmalayici.

    Track ID sayaci ve kayip-kimlik galerisi bu ornege ozeldir -- birden
    fazla kaynagi (orn. iki farkli webcam oturumu) AYNI pipeline uzerinden
    gecirmeyin, her biri icin yeni bir TrackingPipeline() olusturun.
    """

    def __init__(
        self,
        model_weights: str = DEFAULT_WEIGHTS,
        conf_threshold: float = 0.35,
        reid_checkpoint: str | None = DEFAULT_REID_CHECKPOINT,
        fps: float = 30.0,
        person_only: bool = True,
    ):
        self.detector = self._get_detector(model_weights, conf_threshold, person_only)
        self.tracker = AdaptiveTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.3,
            weight_mode="adaptive",
            reid_recovery_enabled=True,
            fps=fps,
            use_fps_scaling=True,
        )
        self.reid_extractor, self.reid_checkpoint_missing = self._get_reid_extractor(reid_checkpoint)
        self.engine = PriorityEngine(zones=[], weights=PRIORITY_WEIGHTS, fps=fps)
        self._t_last = time.time()
        self.frame_count = 0

    @staticmethod
    def _get_detector(model_weights: str, conf_threshold: float, person_only: bool) -> YoloDetector:
        key = (model_weights, conf_threshold, person_only)
        if key not in _detector_cache:
            _detector_cache[key] = YoloDetector(
                weights=model_weights,
                conf_threshold=conf_threshold,
                classes=[0] if person_only else None,  # COCO 0 = person
            )
        return _detector_cache[key]

    @staticmethod
    def _get_reid_extractor(checkpoint_path: str | None) -> tuple[OSNetFeatureExtractor, bool]:
        """Verilen checkpoint bulunamazsa ImageNet-backbone'a duser (main.py'yi
        cokertmemek icin ayni tasarim -- bkz. config.yaml'daki ilgili yorum),
        ama arayuzde bunu SESSIZCE yapmaz -- ikinci deger True ise app.py
        kullaniciya "zayif embedding" uyarisi gosterir."""
        if checkpoint_path in _reid_cache:
            return _reid_cache[checkpoint_path]
        try:
            result = OSNetFeatureExtractor(checkpoint_path=checkpoint_path), False
        except FileNotFoundError:
            result = OSNetFeatureExtractor(checkpoint_path=None), True
        _reid_cache[checkpoint_path] = result
        return result

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR (OpenCV) kare alir, overlay cizilmis BGR kare doner."""
        detections = self.detector.detect(frame_bgr)
        tracks = self.tracker.update(detections, frame_bgr, extractor=self.reid_extractor)

        priority_results = {t.track_id: self.engine.evaluate(t) for t in tracks}

        now = time.time()
        inst_fps = 1.0 / max(now - self._t_last, 1e-6)
        self._t_last = now
        self.frame_count += 1
        visible = sum(1 for t in tracks if getattr(t, "missed", 0) == 0)

        frame_bgr = draw_tracks(frame_bgr, tracks, priority_results)
        frame_bgr = draw_hud(frame_bgr, inst_fps, visible)
        return frame_bgr

    @property
    def unique_track_count(self) -> int:
        """Bu oturumda su ana kadar hic ID verilmis (aktif VEYA aktif olmayan)
        toplam benzersiz track sayisi -- AdaptiveTracker._next_id 1'den baslar
        ve her yeni ID'de artar, hic geriye sarmaz."""
        return self.tracker._next_id - 1


def read_source_fps(cap: cv2.VideoCapture) -> float:
    """Webcam'ler genelde 0 veya guvenilmez bir deger raporlar -- main.py ile
    ayni geri-dusme mantigi (bkz. main.py.run)."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 1 else 30.0
