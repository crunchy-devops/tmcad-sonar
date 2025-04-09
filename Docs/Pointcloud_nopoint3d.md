Great point! To ensure the `ids` in the `PointCloud` class are of type **`long` (64-bit integers)**, we’ll use NumPy’s `int64` (`'i8'`) data type instead of strings.

This is perfect for performance and memory when dealing with large numeric datasets, like LIDAR or GIS point clouds.

---

### ✅ Updated `PointCloud` with `int64` IDs

```python
import numpy as np
from math import sqrt, atan2, degrees

class PointCloud:
    def __init__(self, ids, xs, ys, zs):
        # Ensure all inputs are the same length
        if not (len(ids) == len(xs) == len(ys) == len(zs)):
            raise ValueError("All input arrays must have the same length")

        # Store IDs as NumPy array of int64
        self._ids = np.array(ids, dtype=np.int64)
        self._index_map = {int(id_): i for i, id_ in enumerate(self._ids)}

        # Structured array for points
        self._points = np.zeros(len(ids), dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4')
        ])
        self._points['x'] = xs
        self._points['y'] = ys
        self._points['z'] = zs

    def __len__(self):
        return len(self._points)

    def __getitem__(self, point_id: int):
        idx = self._index_map[point_id]
        p = self._points[idx]
        return (p['x'], p['y'], p['z'])

    def ids(self):
        return self._ids.copy()

    def distance(self, id1: int, id2: int) -> float:
        p1 = self._points[self._index_map[id1]]
        p2 = self._points[self._index_map[id2]]
        dx, dy, dz = p2['x'] - p1['x'], p2['y'] - p1['y'], p2['z'] - p1['z']
        return sqrt(dx**2 + dy**2 + dz**2)

    def slope_percent(self, id1: int, id2: int) -> float:
        p1 = self._points[self._index_map[id1]]
        p2 = self._points[self._index_map[id2]]
        dz = p2['z'] - p1['z']
        horizontal_dist = sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)
        if horizontal_dist == 0:
            return float('inf')
        return (dz / horizontal_dist) * 100

    def bearing(self, id1: int, id2: int) -> float:
        p1 = self._points[self._index_map[id1]]
        p2 = self._points[self._index_map[id2]]
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        angle_rad = atan2(dx, dy)
        angle_deg = degrees(angle_rad)
        return (angle_deg + 360) % 360

    def __repr__(self):
        return f"PointCloud({len(self._points)} points)"
```

---

### ✅ Example usage:

```python
if __name__ == "__main__":
    ids = [1001, 1002, 1003]
    xs = [0.0, 3.0, 5.0]
    ys = [0.0, 4.0, 1.0]
    zs = [0.0, 5.0, 2.0]

    cloud = PointCloud(ids, xs, ys, zs)

    print(cloud)
    print("Distance 1001–1002:", cloud.distance(1001, 1002))
    print("Slope 1001–1002 (%):", cloud.slope_percent(1001, 1002))
    print("Bearing 1001–1002 (°):", cloud.bearing(1001, 1002))
```

---

Let me know if you want to extend this for:
- Bulk calculations (e.g., distance matrix)
- File export/import (CSV, LAS, PLY, etc.)
- Filtering by bounding box or range query

Happy to help!
