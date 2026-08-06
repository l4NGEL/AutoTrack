"""ByteTrack tabanli coklu nesne takip katmani (basitlestirilmis IOU+Kalman implementasyonu)."""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    xyxy: tuple
    cls_name: str
    conf: float
    age: int = 0
    missed: int = 0
    history: list = field(default_factory=list)  # center points across frames, for speed calc

    @property
    def center(self):
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class ByteTracker:
    """IOU eslesmeli, kayip toleransli basit coklu nesne izleyici.

    Ayni imzayi paylasan gercek ByteTrack/DeepSORT paketleriyle degistirilebilir.
    """

    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8, track_buffer: int = 30):
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.track_thresh = track_thresh
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections) -> list[Track]:
        dets = [d for d in detections if d.conf >= self.track_thresh]

        if not self.tracks:
            for det in dets:
                self.tracks.append(self._new_track(det))
            return self.tracks

        if not dets:
            for t in self.tracks:
                t.missed += 1
            self._prune()
            return self.tracks

        cost = np.zeros((len(self.tracks), len(dets)))
        for i, t in enumerate(self.tracks):
            for j, d in enumerate(dets):
                cost[i, j] = 1.0 - iou(t.xyxy, d.xyxy)

        row_idx, col_idx = linear_sum_assignment(cost)

        matched_tracks, matched_dets = set(), set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] <= (1.0 - 0.1):  # IOU >= 0.1 to count as match
                track = self.tracks[r]
                det = dets[c]
                track.history.append(track.center)
                track.xyxy = det.xyxy
                track.cls_name = det.cls_name
                track.conf = det.conf
                track.age += 1
                track.missed = 0
                matched_tracks.add(r)
                matched_dets.add(c)

        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.missed += 1

        for j, d in enumerate(dets):
            if j not in matched_dets:
                self.tracks.append(self._new_track(d))

        self._prune()
        return self.tracks

    def _new_track(self, det) -> Track:
        track = Track(track_id=self._next_id, xyxy=det.xyxy, cls_name=det.cls_name, conf=det.conf)
        self._next_id += 1
        return track

    def _prune(self):
        self.tracks = [t for t in self.tracks if t.missed <= self.track_buffer]
