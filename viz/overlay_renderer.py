"""Canli video uzerine kutu / ID / oncelik skoru / hassas bolge overlay cizimi."""
import cv2
import numpy as np

from decision.zone_rules import Zone

BOX_COLOR_LOW = (80, 200, 80)
BOX_COLOR_MED = (0, 200, 255)
BOX_COLOR_HIGH = (0, 0, 255)
ZONE_COLOR = (255, 120, 0)


def _color_for_score(score: float):
    if score >= 0.7:
        return BOX_COLOR_HIGH
    if score >= 0.4:
        return BOX_COLOR_MED
    return BOX_COLOR_LOW


def draw_zones(frame, zones: list[Zone]):
    for zone in zones:
        pts = zone.polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=ZONE_COLOR, thickness=2)
        x, y = zone.polygon[0]
        cv2.putText(frame, zone.name, (int(x), int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ZONE_COLOR, 1)
    return frame


def draw_tracks(frame, tracks, priority_results: dict):
    for track in tracks:
        result = priority_results.get(track.track_id)
        score = result.score if result else 0.0
        color = _color_for_score(score)

        x1, y1, x2, y2 = [int(v) for v in track.xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {track.track_id} {track.cls_name} P={score:.2f}"
        bucket = getattr(track, "confidence_bucket", None)
        if bucket is not None:
            label += f" [{bucket.value}]"
        cv2.putText(frame, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if result and result.reasons:
            reason_text = ", ".join(result.reasons)
            cv2.putText(frame, reason_text, (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return frame


def draw_hud(frame, fps: float, num_tracks: int):
    text = f"FPS: {fps:.1f}  Hedefler: {num_tracks}"
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame
