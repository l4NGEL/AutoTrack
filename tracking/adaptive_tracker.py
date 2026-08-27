"""IOU + hareket (Kalman) + gorunum (Re-ID) ipuclarini adaptif agirliklarla
birlestiren coklu hedef izleyici.

Sabit agirlikli ("fixed") ve adaptif agirlikli ("adaptive") modlari
destekler; ikisi de ayni birlesim/atama koduna girer, boylece ablation
calismasinda (evaluation/run_ablation.py) tek bir tracker sinifi farkli
agirlik stratejileriyle adil sekilde karsilastirilabilir.

Birlesik skor:  S = alpha * S_iou + beta * S_motion + gamma * S_reid
                (alpha + beta + gamma = 1)

Adaptif mod, alpha/beta/gamma'yi HER TRACK icin su sinyallere gore yeniden
hesaplar (bkz. compute_adaptive_weights):

  - occlusion_signal: bu track'in tahmini kutusunun, baska bir track'in
    tahmini kutusuyla en yuksek IOU'su. Yuksekse Re-ID (gamma) agirligi
    artar, IOU (alpha) agirligi azalir -- kutular ortustugunde geometri tek
    basina kimligi ayirt edemez.
  - motion_magnitude: Kalman hiz vektorunun buyuklugu, kare kosegenine
    normalize edilir. Yuksekse (hizli hareket) hareket (beta) agirligi
    artar, IOU (alpha) agirligi azalir -- hizli hareket eden hedeflerde
    ardisik kareler arasi kutu ortusmesi dusuk olabilir.
  - crop_quality: tespit guveni ve kutu alaninin referans (ilk gorulen)
    alana orani. Dusukse (kucuk/belirsiz kirpim) Re-ID (gamma) agirligi
    azalir -- kucuk/bulanik kirpimdan cikan embedding guvenilmezdir.

Kayip kimlik kurtarma (lost-track gallery):
Bir track `track_buffer` kare boyunca eslesmezse silinir -- ama silinmeden
once (person sinifiysa ve embedding'i varsa) `lost_gallery`'ye tasinir ve
orada `reid_recovery_window` kare daha "aday kimlik" olarak bekler. Bu sure
icinde eslesmemis (yeni track adayi olan) bir tespit gelirse, once bu
galeriye karsi Re-ID benzerligi kontrol edilir (`reid_recovery_thresh` --
match_thresh'ten daha siki, cunku mekansal/hareket dogrulamasi yok, sadece
gorunume guveniliyor + kaba bir mesafe siniri). Yeterince benzerse YENI bir
ID verilmez, eski track_id geri canlandirilir. Bu, DeepSORT'un "silinen
track'leri bir sure daha eslestirme adayi tutma" fikrinin basitlestirilmis
hali -- 2026-08-27'de gercek bir demo videosunda gozlenen "kisi kayboluyor,
birkac kare sonra yeni ID ile geri geliyor" sorununa dogrudan cevap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from reid.feature_extractor import cosine_similarity
from tracking.bytetrack_wrapper import iou
from tracking.kalman_motion import KalmanBoxTracker

REID_ELIGIBLE_CLASSES = {"person"}

DEFAULT_BASE_WEIGHTS = (0.4, 0.3, 0.3)  # alpha (IOU), beta (hareket), gamma (Re-ID)
OCCLUSION_IOU_THRESH = 0.15
FAST_MOTION_NORM_THRESH = 0.02  # frame kosegenine oranla hiz
MIN_CROP_QUALITY_CONF = 0.55
MIN_CROP_AREA_RATIO = 0.15  # referans (ilk gorulen) alana oran

# Kayip kimlik kurtarma (lost-track gallery) icin varsayilanlar
DEFAULT_REID_RECOVERY_WINDOW = 60  # kare -- track_buffer'dan sonra kimlik ne kadar sure "aday" kalir
DEFAULT_REID_RECOVERY_THRESH = 0.75  # ham cosine benzerligi (mapped [0,1] degil) -- match_thresh'ten siki
MAX_RECOVERY_DIST_RATIO = 0.2  # kare kosegenine oran -- bunun otesindeki "benzer" eslesmeler reddedilir
                                # (2026-08-27: 0.5 idi, gercek bir yanlis-birlestirme bulundu ve sikilastirildi -- bkz. README "Faz 2.2")

# Tek karede fiziksel olarak imkansiz "ziplamalari" onleyen sert mesafe siniri.
# Adaptif agirliklandirma occlusion/hiz sinyaline gore alpha'yi (IOU) cok
# dusurdugunde, guclu ama YANLIS bir Re-ID/hareket skoru teorik olarak
# tamamen alakasiz bir tespitle eslesmeyi "kazanabilir" -- kombine skor
# yeterince yuksek gorunse bile boyle bir eslesme fiziksel olarak imkansizdir.
# Bu, agirliklardan BAGIMSIZ bir on-eleme: mesafe asilirsa skor ne olursa
# olsun eslesme reddedilir.
MAX_ASSOCIATION_DIST_RATIO = 0.15  # kare kosegenine oran, TEK kare icin


class Confidence(Enum):
    STABLE = "stable"
    UNCERTAIN = "uncertain"
    LOST_CANDIDATE = "lost_candidate"


@dataclass
class AdaptiveTrack:
    track_id: int
    xyxy: tuple
    cls_name: str
    conf: float
    kalman: KalmanBoxTracker
    embedding: np.ndarray | None = None
    ref_area: float = 0.0
    age: int = 0
    missed: int = 0
    history: list = field(default_factory=list)
    det_conf_ema: float = 0.0
    motion_conf: float = 1.0
    reid_conf: float = 0.0
    weights_used: tuple = DEFAULT_BASE_WEIGHTS
    last_confirmed_xyxy: tuple | None = None  # son GERCEK eslesmedeki konum -- xyxy'nin aksine Kalman ekstrapolasyonuyla surunmez

    @property
    def center(self):
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def track_confidence(self) -> float:
        has_reid = self.embedding is not None and self.cls_name in REID_ELIGIBLE_CLASSES
        if has_reid:
            return 0.4 * self.det_conf_ema + 0.3 * self.motion_conf + 0.3 * self.reid_conf
        return 0.6 * self.det_conf_ema + 0.4 * self.motion_conf

    @property
    def confidence_bucket(self) -> Confidence:
        c = self.track_confidence
        if c >= 0.75:
            return Confidence.STABLE
        if c >= 0.5:
            return Confidence.UNCERTAIN
        return Confidence.LOST_CANDIDATE


@dataclass
class LostTrack:
    """track_buffer asilip silinen bir track'in Re-ID kurtarma icin bir
    sure daha sakladigimiz ozeti. Kalman/history gibi hareket durumunu
    TASIMAZ -- o zaten gecersiz hale gelmistir (bkz. AdaptiveTracker._revive_track)."""

    track_id: int
    embedding: np.ndarray
    last_xyxy: tuple
    cls_name: str
    age: int = 0  # kayip galeride kac karedir bekliyor


def compute_adaptive_weights(
    track: AdaptiveTrack,
    track_index: int,
    predicted_boxes: list,
    frame_diag: float,
    base_weights=DEFAULT_BASE_WEIGHTS,
) -> tuple:
    alpha, beta, gamma = base_weights

    if track.cls_name not in REID_ELIGIBLE_CLASSES:
        total = alpha + beta
        return (alpha / total, beta / total, 0.0) if total > 0 else (0.5, 0.5, 0.0)

    occlusion = max(
        (iou(track.xyxy, box) for i, box in enumerate(predicted_boxes) if i != track_index),
        default=0.0,
    )
    if occlusion > OCCLUSION_IOU_THRESH:
        shift = min(0.2, 0.15 * (occlusion / OCCLUSION_IOU_THRESH))
        alpha -= shift
        gamma += shift

    motion_norm = track.kalman.velocity_magnitude / max(frame_diag, 1.0)
    if motion_norm > FAST_MOTION_NORM_THRESH:
        shift = min(0.15, 0.15 * (motion_norm / FAST_MOTION_NORM_THRESH - 1.0))
        alpha -= shift
        beta += shift

    area = (track.xyxy[2] - track.xyxy[0]) * (track.xyxy[3] - track.xyxy[1])
    area_ratio = area / track.ref_area if track.ref_area > 0 else 1.0
    low_quality = track.conf < MIN_CROP_QUALITY_CONF or area_ratio < MIN_CROP_AREA_RATIO
    if low_quality and gamma > 0:
        shift = gamma * 0.6
        gamma -= shift
        alpha += shift * 0.5
        beta += shift * 0.5

    alpha, beta, gamma = max(alpha, 0.0), max(beta, 0.0), max(gamma, 0.0)
    total = alpha + beta + gamma
    return (alpha / total, beta / total, gamma / total) if total > 0 else (1.0, 0.0, 0.0)


class AdaptiveTracker:
    """weight_mode='adaptive' -> compute_adaptive_weights kullanir (her track
    icin dinamik alpha/beta/gamma). weight_mode='fixed' -> base_weights hep
    ayni kalir (ablation calismasindaki A-D varyantlari icin)."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.3,
        track_buffer: int = 30,
        weight_mode: str = "adaptive",
        base_weights: tuple = DEFAULT_BASE_WEIGHTS,
        embedding_momentum: float = 0.9,
        reid_recovery_enabled: bool = True,
        reid_recovery_window: int = DEFAULT_REID_RECOVERY_WINDOW,
        reid_recovery_thresh: float = DEFAULT_REID_RECOVERY_THRESH,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh  # min birlesik skor (0-1); altinda eslesme reddedilir
        self.track_buffer = track_buffer
        self.weight_mode = weight_mode
        self.base_weights = tuple(base_weights)
        self.embedding_momentum = embedding_momentum
        self.reid_recovery_enabled = reid_recovery_enabled
        self.reid_recovery_window = reid_recovery_window
        self.reid_recovery_thresh = reid_recovery_thresh

        self.tracks: list[AdaptiveTrack] = []
        self.lost_gallery: list[LostTrack] = []
        self._next_id = 1

    def update(self, detections, frame, extractor=None) -> list[AdaptiveTrack]:
        dets = [d for d in detections if d.conf >= self.track_thresh]
        h, w = frame.shape[:2]
        frame_diag = float(np.hypot(w, h))

        det_embeddings = self._extract_embeddings(dets, frame, extractor)

        predicted_boxes = []
        for t in self.tracks:
            t.xyxy = t.kalman.predict()
            predicted_boxes.append(t.xyxy)

        if not self.tracks:
            for det, emb in zip(dets, det_embeddings):
                revived = self._try_revive(det, emb, frame_diag)
                self.tracks.append(revived if revived is not None else self._new_track(det, emb))
            self._prune()  # tracks henuz taze (missed=0) -- bu sadece lost_gallery'yi yaslandirir
            return self.tracks

        if not dets:
            for t in self.tracks:
                t.missed += 1
                t.motion_conf *= 0.9
            self._prune()
            return self.tracks

        cost = self._cost_matrix(dets, det_embeddings, predicted_boxes, frame_diag)
        row_idx, col_idx = linear_sum_assignment(cost)

        matched_tracks, matched_dets = set(), set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] <= (1.0 - self.match_thresh):
                self._assign(self.tracks[r], dets[c], det_embeddings[c])
                matched_tracks.add(r)
                matched_dets.add(c)

        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.missed += 1
                t.motion_conf *= 0.85

        for j, det in enumerate(dets):
            if j not in matched_dets:
                revived = self._try_revive(det, det_embeddings[j], frame_diag)
                self.tracks.append(revived if revived is not None else self._new_track(det, det_embeddings[j]))

        self._prune()
        return self.tracks

    def _extract_embeddings(self, dets, frame, extractor):
        if extractor is None:
            return [None] * len(dets)
        person_idx = [i for i, d in enumerate(dets) if d.cls_name in REID_ELIGIBLE_CLASSES]
        embeddings = [None] * len(dets)
        if person_idx:
            boxes = [dets[i].xyxy for i in person_idx]
            feats = extractor.extract(frame, boxes)
            for idx, i in enumerate(person_idx):
                embeddings[i] = feats[idx]
        return embeddings

    def _cost_matrix(self, dets, det_embeddings, predicted_boxes, frame_diag):
        max_dist = MAX_ASSOCIATION_DIST_RATIO * frame_diag
        cost = np.ones((len(self.tracks), len(dets)))
        for i, t in enumerate(self.tracks):
            if self.weight_mode == "adaptive":
                weights = compute_adaptive_weights(t, i, predicted_boxes, frame_diag, self.base_weights)
            else:
                weights = self._masked_fixed_weights(t)
            alpha, beta, gamma = weights
            tcx, tcy = t.center
            for j, d in enumerate(dets):
                dcx, dcy = (d.xyxy[0] + d.xyxy[2]) / 2, (d.xyxy[1] + d.xyxy[3]) / 2
                if float(np.hypot(dcx - tcx, dcy - tcy)) > max_dist:
                    continue  # skor ne olursa olsun: tek karede bu kadar uzaga ziplamak fiziksel olarak imkansiz
                s_iou = iou(t.xyxy, d.xyxy)
                s_motion = self._motion_score(t, d, frame_diag)
                s_reid = self._reid_score(t, det_embeddings[j])
                score = alpha * s_iou + beta * s_motion + gamma * s_reid
                cost[i, j] = 1.0 - score
            t.weights_used = weights
        return cost

    def _masked_fixed_weights(self, track):
        alpha, beta, gamma = self.base_weights
        if track.cls_name not in REID_ELIGIBLE_CLASSES:
            total = alpha + beta
            return (alpha / total, beta / total, 0.0) if total > 0 else (0.5, 0.5, 0.0)
        return (alpha, beta, gamma)

    @staticmethod
    def _motion_score(track, det, frame_diag) -> float:
        px, py = track.center
        dx, dy = (det.xyxy[0] + det.xyxy[2]) / 2, (det.xyxy[1] + det.xyxy[3]) / 2
        dist = float(np.hypot(dx - px, dy - py))
        return float(np.exp(-dist / max(0.08 * frame_diag, 1.0)))

    @staticmethod
    def _reid_score(track, det_embedding) -> float:
        if track.embedding is None or det_embedding is None:
            return 0.0
        sim = cosine_similarity(track.embedding, det_embedding)
        return max(0.0, (sim + 1.0) / 2.0)

    def _assign(self, track: AdaptiveTrack, det, embedding) -> None:
        track.history.append(track.center)
        track.kalman.update(det.xyxy)
        track.xyxy = det.xyxy
        track.last_confirmed_xyxy = det.xyxy
        track.cls_name = det.cls_name
        track.conf = det.conf
        track.age += 1
        track.missed = 0
        track.det_conf_ema = 0.7 * track.det_conf_ema + 0.3 * det.conf

        max_innov = 5.0
        track.motion_conf = float(np.clip(1.0 - track.kalman.last_innovation_norm / max_innov, 0.0, 1.0))

        if embedding is not None:
            if track.embedding is None:
                track.embedding = embedding
            else:
                m = self.embedding_momentum
                track.embedding = m * track.embedding + (1 - m) * embedding
                track.embedding = track.embedding / (np.linalg.norm(track.embedding) + 1e-8)
            track.reid_conf = max(0.0, (cosine_similarity(track.embedding, embedding) + 1.0) / 2.0)

    def _new_track(self, det, embedding) -> AdaptiveTrack:
        kalman = KalmanBoxTracker(det.xyxy)
        area = (det.xyxy[2] - det.xyxy[0]) * (det.xyxy[3] - det.xyxy[1])
        track = AdaptiveTrack(
            track_id=self._next_id,
            xyxy=det.xyxy,
            cls_name=det.cls_name,
            conf=det.conf,
            kalman=kalman,
            embedding=embedding,
            ref_area=area,
            det_conf_ema=det.conf,
            motion_conf=1.0,
            reid_conf=0.0,
            last_confirmed_xyxy=det.xyxy,
        )
        self._next_id += 1
        return track

    def _try_revive(self, det, embedding, frame_diag: float) -> AdaptiveTrack | None:
        """lost_gallery'de bu tespitle yeterince benzer bir kimlik varsa
        eski track_id'yi geri canlandirir; yoksa None doner (cagiran taraf
        normal sekilde yeni bir track acar)."""
        if not self.reid_recovery_enabled or not self.lost_gallery:
            return None
        if embedding is None or det.cls_name not in REID_ELIGIBLE_CLASSES:
            return None

        dcx, dcy = (det.xyxy[0] + det.xyxy[2]) / 2, (det.xyxy[1] + det.xyxy[3]) / 2
        max_dist = MAX_RECOVERY_DIST_RATIO * frame_diag

        best_idx, best_sim = -1, -1.0
        for idx, lt in enumerate(self.lost_gallery):
            if lt.cls_name != det.cls_name:
                continue
            lcx, lcy = (lt.last_xyxy[0] + lt.last_xyxy[2]) / 2, (lt.last_xyxy[1] + lt.last_xyxy[3]) / 2
            if float(np.hypot(dcx - lcx, dcy - lcy)) > max_dist:
                continue  # cok uzak -- gorunum benzerligi tek basina yetmez
            sim = cosine_similarity(lt.embedding, embedding)  # ham [-1,1], (0,1]'e esitlenmemis
            if sim > best_sim:
                best_idx, best_sim = idx, sim

        if best_idx == -1 or best_sim < self.reid_recovery_thresh:
            return None

        lost = self.lost_gallery.pop(best_idx)
        return self._revive_track(lost, det, embedding)

    def _revive_track(self, lost: LostTrack, det, embedding) -> AdaptiveTrack:
        # Eski Kalman/history durumu kayip sure boyunca gecersiz hale
        # gelmistir (bkz. LostTrack docstring) -- hareket sifirdan baslar,
        # ama kimlik (track_id) ve gorunum (embedding, EMA ile harmanlanmis)
        # korunur.
        kalman = KalmanBoxTracker(det.xyxy)
        area = (det.xyxy[2] - det.xyxy[0]) * (det.xyxy[3] - det.xyxy[1])
        m = self.embedding_momentum
        blended = m * lost.embedding + (1 - m) * embedding
        blended = blended / (np.linalg.norm(blended) + 1e-8)
        return AdaptiveTrack(
            track_id=lost.track_id,
            xyxy=det.xyxy,
            cls_name=det.cls_name,
            conf=det.conf,
            kalman=kalman,
            embedding=blended,
            ref_area=area,
            det_conf_ema=det.conf,
            motion_conf=1.0,
            reid_conf=max(0.0, (cosine_similarity(lost.embedding, embedding) + 1.0) / 2.0),
            last_confirmed_xyxy=det.xyxy,
        )

    def _prune(self) -> None:
        still_active, newly_lost = [], []
        for t in self.tracks:
            if t.missed <= self.track_buffer:
                still_active.append(t)
            elif self.reid_recovery_enabled and t.embedding is not None and t.cls_name in REID_ELIGIBLE_CLASSES:
                # last_xyxy = son GERCEK tespit (last_confirmed_xyxy), t.xyxy DEGIL --
                # t.xyxy burada zaten track_buffer kadar Kalman ekstrapolasyonuyla
                # surunmus olabilir, kurtarma mesafe kontrolu icin guvenilmez.
                ref_box = t.last_confirmed_xyxy or t.xyxy
                newly_lost.append(LostTrack(track_id=t.track_id, embedding=t.embedding, last_xyxy=ref_box, cls_name=t.cls_name))
        self.tracks = still_active

        for lt in self.lost_gallery:
            lt.age += 1
        self.lost_gallery = [lt for lt in self.lost_gallery if lt.age <= self.reid_recovery_window] + newly_lost
