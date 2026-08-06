"""Acikla nabilir kural tabanli oncelik skorlama motoru.

Her hedef icin skor = w_speed * hiz_skoru + w_zone * bolge_skoru + w_dwell * bekleme_skoru
Her bilesen [0, 1] araligina normalize edilir ve nedeni ile birlikte raporlanir.
"""
from dataclasses import dataclass, field
import time

from decision.zone_rules import Zone

MAX_EXPECTED_SPEED_PX_PER_FRAME = 40.0
MAX_DWELL_SECONDS = 15.0


@dataclass
class PriorityResult:
    track_id: int
    score: float
    reasons: list[str] = field(default_factory=list)


class PriorityEngine:
    def __init__(self, zones: list[Zone], weights: dict, fps: float = 30.0):
        self.zones = zones
        self.weights = weights
        self.fps = fps
        self._zone_entry_time: dict[int, float] = {}

    def evaluate(self, track) -> PriorityResult:
        reasons = []

        speed_score = self._speed_score(track)
        if speed_score > 0.5:
            reasons.append(f"yuksek hiz ({speed_score:.2f})")

        zone_name, zone_score = self._zone_score(track)
        if zone_score > 0:
            reasons.append(f"hassas bolgede: {zone_name}")

        dwell_score = self._dwell_score(track, in_zone=zone_score > 0)
        if dwell_score > 0.5:
            reasons.append(f"uzun sure bolgede bekliyor ({dwell_score:.2f})")

        score = (
            self.weights.get("speed", 0.4) * speed_score
            + self.weights.get("zone", 0.4) * zone_score
            + self.weights.get("dwell_time", 0.2) * dwell_score
        )

        if not reasons:
            reasons.append("normal aktivite")

        return PriorityResult(track_id=track.track_id, score=round(score, 3), reasons=reasons)

    def _speed_score(self, track) -> float:
        if len(track.history) < 2:
            return 0.0
        (x1, y1), (x2, y2) = track.history[-2], track.history[-1]
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return min(dist / MAX_EXPECTED_SPEED_PX_PER_FRAME, 1.0)

    def _zone_score(self, track) -> tuple[str, float]:
        for zone in self.zones:
            if zone.contains(track.center):
                return zone.name, 1.0
        return "", 0.0

    def _dwell_score(self, track, in_zone: bool) -> float:
        now = time.time()
        if not in_zone:
            self._zone_entry_time.pop(track.track_id, None)
            return 0.0

        entered = self._zone_entry_time.setdefault(track.track_id, now)
        dwell = now - entered
        return min(dwell / MAX_DWELL_SECONDS, 1.0)
