Perfect! Here's an updated version of the immutable `PointCloud` class with:

### ✅ **New Features**
1. **Uniqueness enforcement** for `id` — duplicates are not allowed.
2. **Serialization methods**:
   - `to_dict()` – returns list of dicts.
   - `to_json()` – JSON string representation.
   - `to_ply()` – optional `.ply` file export (if needed).
3. Efficient and still immutable — uses `dataclass(frozen=True)`.

---

### 🧱 Final Version

```python
from dataclasses import dataclass, field
import numpy as np
import json
from typing import Optional, List


@dataclass(frozen=True)
class PointCloud:
    points: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=PointCloud._dtype()))

    @staticmethod
    def _dtype():
        return np.dtype([
            ('id', np.int64),
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32)
        ])

    def add_point(self, point_id: int, x: float, y: float, z: float) -> 'PointCloud':
        if self.get_point_by_id(point_id) is not None:
            raise ValueError(f"Point ID {point_id} already exists.")
        new_point = np.array([(point_id, x, y, z)], dtype=self._dtype())
        new_points = np.append(self.points, new_point)
        return PointCloud(new_points)

    def remove_point_by_id(self, point_id: int) -> 'PointCloud':
        filtered = self.points[self.points['id'] != point_id]
        return PointCloud(filtered)

    def get_point_by_id(self, point_id: int) -> Optional[np.void]:
        match = self.points[self.points['id'] == point_id]
        return match[0] if match.size > 0 else None

    def get_all_points(self) -> np.ndarray:
        return self.points.copy()

    def to_dict(self) -> List[dict]:
        return [
            {"id": int(p["id"]), "x": float(p["x"]), "y": float(p["y"]), "z": float(p["z"])}
            for p in self.points
        ]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_ply(self, filename: str) -> None:
        header = f"""ply
format ascii 1.0
element vertex {len(self.points)}
property int64 id
property float x
property float y
property float z
end_header
"""
        with open(filename, 'w') as f:
            f.write(header)
            for p in self.points:
                f.write(f"{p['id']} {p['x']} {p['y']} {p['z']}\n")

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"<Immutable PointCloud with {len(self.points)} points>"
```

---

### ✅ Example Usage

```python
pc = PointCloud()
pc = pc.add_point(1, 1.0, 2.0, 3.0)
pc = pc.add_point(2, 4.0, 5.0, 6.0)

print(pc.to_json())
pc.to_ply("pointcloud.ply")

print(pc.get_point_by_id(1))  # Access single point
```

---

Let me know if you’d like support for `numpy.save/load`, `.pcd` format, or integration with libraries like `open3d` or `pandas`.
