"""FPS, tespit sayisi ve karar gecmisini JSONL olarak loglayan performans modulu."""
import json
import time
from pathlib import Path


class PerformanceLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._frame_count = 0
        self._start_time = time.time()
        self._last_tick = self._start_time

    def tick(self) -> float:
        now = time.time()
        elapsed = now - self._last_tick
        self._last_tick = now
        self._frame_count += 1
        return 1.0 / elapsed if elapsed > 0 else 0.0

    @property
    def avg_fps(self) -> float:
        elapsed = time.time() - self._start_time
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    def log_frame(self, fps: float, num_detections: int, priority_results: list):
        entry = {
            "ts": time.time(),
            "fps": round(fps, 2),
            "num_detections": num_detections,
            "decisions": [
                {"track_id": r.track_id, "score": r.score, "reasons": r.reasons}
                for r in priority_results
            ],
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
