"""Hassas bolge tanimlari ve nokta-poligon icinde mi kontrolu."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Zone:
    name: str
    polygon: np.ndarray  # shape (N, 2)

    @classmethod
    def from_config(cls, name: str, points: list[list[float]]) -> "Zone":
        return cls(name=name, polygon=np.array(points, dtype=np.float32))

    def contains(self, point: tuple[float, float]) -> bool:
        # cv2.pointPolygonTest ile ayni sonucu veren bagimsizlastirilmis ray-casting
        x, y = point
        poly = self.polygon
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside


def load_zones(zone_configs: list[dict]) -> list[Zone]:
    return [Zone.from_config(z["name"], z["polygon"]) for z in zone_configs]
