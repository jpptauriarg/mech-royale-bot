"""Campus occupancy grid, line of sight, and coarse paths. Standard library only."""
from __future__ import annotations

import heapq
import json
import math
import os
from array import array
from typing import Dict, List, Optional, Sequence, Tuple

Vec = Tuple[float, float]
_CACHE: Dict[str, "Arena"] = {}


def get_arena(map_dir: str) -> "Arena":
    key = os.path.abspath(map_dir)
    arena = _CACHE.get(key)
    if arena is None:
        arena = Arena(key)
        _CACHE[key] = arena
    return arena


def _pip(x: float, y: float, ring: Sequence[float]) -> bool:
    n = len(ring) // 2
    if n < 3:
        return False
    inside = False
    xj = ring[-2]
    yj = ring[-1]
    for i in range(n):
        xi = ring[2 * i]
        yi = ring[2 * i + 1]
        if (yi > y) != (yj > y):
            denom = yj - yi
            if denom != 0.0 and x < (xj - xi) * (y - yi) / denom + xi:
                inside = not inside
        xj, yj = xi, yi
    return inside


class Arena:
    """8 m occupancy grid. A cell is blocked when a solid overlaps a walking body."""

    def __init__(self, map_dir: str, cell: float = 8.0):
        self.ok = False
        self.cell = cell
        self.minx = 0.0
        self.miny = 0.0
        self.nx = 0
        self.ny = 0
        self.blocked = bytearray()
        self.cover = bytearray()
        self.bounds = (-1e9, 1e9, -1e9, 1e9)
        self._terrain: Optional[array] = None
        self.t_minx = self.t_miny = 0.0
        self.t_step = 1.0
        self.t_nx = self.t_ny = 0
        self.t_scale = 0.01
        try:
            self._load(map_dir)
            self.ok = True
        except Exception:
            self.ok = False

    def _load(self, map_dir: str) -> None:
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            world = json.load(f)
        b = world["bounds"]
        self.bounds = (b["minX"], b["maxX"], b["minY"], b["maxY"])
        self._load_terrain(map_dir, world["terrain"])
        self.minx = math.floor(b["minX"] / self.cell) * self.cell - self.cell
        self.miny = math.floor(b["minY"] / self.cell) * self.cell - self.cell
        maxx = b["maxX"] + self.cell * 2
        maxy = b["maxY"] + self.cell * 2
        self.nx = int((maxx - self.minx) / self.cell) + 1
        self.ny = int((maxy - self.miny) / self.cell) + 1
        self.blocked = bytearray(self.nx * self.ny)
        self._raster(world["prisms"])
        self._close_border()
        self._mark_cover()

    def _load_terrain(self, map_dir: str, t: dict) -> None:
        n = int(t["nx"] * t["ny"])
        data = array("h")
        with open(os.path.join(map_dir, "v2", t["file"]), "rb") as f:
            data.fromfile(f, n)
        self._terrain = data
        self.t_minx = float(t["minX"])
        self.t_miny = float(t["minY"])
        self.t_step = float(t["step"])
        self.t_nx = int(t["nx"])
        self.t_ny = int(t["ny"])
        self.t_scale = float(t.get("scale", 0.01))

    def height(self, x: float, y: float) -> float:
        if self._terrain is None:
            return 0.0
        fx = (x - self.t_minx) / self.t_step
        fy = (y - self.t_miny) / self.t_step
        if fx < 0 or fy < 0 or fx > self.t_nx - 1 or fy > self.t_ny - 1:
            return 0.0
        i = int(fx)
        j = int(fy)
        tx = fx - i
        ty = fy - j
        i2 = min(i + 1, self.t_nx - 1)
        j2 = min(j + 1, self.t_ny - 1)
        nx = self.t_nx

        def h(ii: int, jj: int) -> float:
            return self._terrain[jj * nx + ii] * self.t_scale  # type: ignore[index]

        h00 = h(i, j)
        h10 = h(i2, j)
        h01 = h(i, j2)
        h11 = h(i2, j2)
        return (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty

    def _idx(self, i: int, j: int) -> int:
        return j * self.nx + i

    def cell_at(self, x: float, y: float) -> Tuple[int, int]:
        i = int((x - self.minx) / self.cell)
        j = int((y - self.miny) / self.cell)
        if i < 0:
            i = 0
        elif i >= self.nx:
            i = self.nx - 1
        if j < 0:
            j = 0
        elif j >= self.ny:
            j = self.ny - 1
        return i, j

    def in_bounds(self, x: float, y: float, pad: float = 2.0) -> bool:
        minx, maxx, miny, maxy = self.bounds
        return minx + pad <= x <= maxx - pad and miny + pad <= y <= maxy - pad

    def is_blocked_cell(self, i: int, j: int) -> bool:
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return True
        return self.blocked[self._idx(i, j)] != 0

    def blocked_at(self, x: float, y: float) -> bool:
        if not self.ok:
            return False
        if not self.in_bounds(x, y, 1.0):
            return True
        i, j = self.cell_at(x, y)
        return self.is_blocked_cell(i, j)

    def has_cover(self, x: float, y: float) -> bool:
        if not self.ok or self.blocked_at(x, y):
            return False
        i, j = self.cell_at(x, y)
        return self.cover[self._idx(i, j)] != 0

    def _sample_blocked(self, prepared, buckets, x: float, y: float) -> bool:
        if not self.in_bounds(x, y, 1.0):
            return True
        tz = self.height(x, y)
        i, j = self.cell_at(x, y)
        for pi in buckets.get((i, j), ()):
            p, ring, holes, minx, maxx, miny, maxy = prepared[pi]
            if x < minx or x > maxx or y < miny or y > maxy:
                continue
            base = p["base"]
            top = p["top"]
            # Buried, or a deck you can step onto, does not stop a walker.
            if top <= tz + 1.2:
                continue
            # Clearance above the head (mech height 9 m, small tolerance).
            if base >= tz + 8.6:
                continue
            if not _pip(x, y, ring):
                continue
            if any(_pip(x, y, h) for h in holes):
                continue
            return True
        return False

    def _raster(self, prisms: list) -> None:
        prepared = []
        buckets: Dict[Tuple[int, int], List[int]] = {}
        for pi, p in enumerate(prisms):
            ring = p["ring"]
            xs = ring[0::2]
            ys = ring[1::2]
            if not xs:
                continue
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            prepared.append((p, ring, p.get("holes") or [], minx, maxx, miny, maxy))
            i0, j0 = self.cell_at(minx, miny)
            i1, j1 = self.cell_at(maxx, maxy)
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    buckets.setdefault((i, j), []).append(pi)
        # Four samples per cell so thin buildings are not skipped.
        offs = ((0.0, 0.0), (-2.2, -2.2), (2.2, -2.2), (-2.2, 2.2), (2.2, 2.2))
        for j in range(self.ny):
            cy = self.miny + (j + 0.5) * self.cell
            for i in range(self.nx):
                cx = self.minx + (i + 0.5) * self.cell
                hit = False
                for ox, oy in offs:
                    if self._sample_blocked(prepared, buckets, cx + ox, cy + oy):
                        hit = True
                        break
                if hit:
                    self.blocked[self._idx(i, j)] = 1

    def _close_border(self) -> None:
        minx, maxx, miny, maxy = self.bounds
        for j in range(self.ny):
            y = self.miny + (j + 0.5) * self.cell
            for i in range(self.nx):
                x = self.minx + (i + 0.5) * self.cell
                if x < minx + 4 or x > maxx - 4 or y < miny + 4 or y > maxy - 4:
                    self.blocked[self._idx(i, j)] = 1

    def _mark_cover(self) -> None:
        self.cover = bytearray(self.nx * self.ny)
        for j in range(self.ny):
            for i in range(self.nx):
                if self.blocked[self._idx(i, j)]:
                    continue
                for dj in (-1, 0, 1):
                    for di in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        if self.is_blocked_cell(i + di, j + dj):
                            self.cover[self._idx(i, j)] = 1
                            dj = 99
                            break
                    if dj == 99:
                        break

    def nearest_open(self, x: float, y: float, rad_cells: int = 6) -> Vec:
        if not self.ok or not self.blocked_at(x, y):
            return x, y
        i0, j0 = self.cell_at(x, y)
        for r in range(1, rad_cells + 1):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    i, j = i0 + di, j0 + dj
                    if not self.is_blocked_cell(i, j):
                        return self.minx + (i + 0.5) * self.cell, self.miny + (j + 0.5) * self.cell
        return x, y

    def los(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Ground line of sight. Cells touching the two endpoints are ignored."""
        if not self.ok:
            return True
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 6.0:
            return True
        step = self.cell * 0.45
        n = max(2, int(dist / step))
        # Skip ~1 cell at each end so hugging a wall still counts as seeing out of it.
        start = max(1, int(self.cell / step))
        stop = n - start
        if stop <= start:
            return True
        for k in range(start, stop):
            t = k / n
            if self.blocked_at(x0 + dx * t, y0 + dy * t):
                return False
        return True

    def avoid(self, x: float, y: float, mx: float, my: float) -> Vec:
        """Turn a desired direction off an upcoming wall."""
        mag = math.hypot(mx, my)
        if mag < 1e-6 or not self.ok:
            return mx, my
        ux, uy = mx / mag, my / mag
        if not self.blocked_at(x + ux * 9.0, y + uy * 9.0) and not self.blocked_at(x + ux * 16.0, y + uy * 16.0):
            return mx, my
        base = math.atan2(uy, ux)
        best: Optional[Tuple[float, float, float]] = None
        for deg in (35, -35, 60, -60, 95, -95, 130, -130, 170, -170):
            a = base + deg * math.pi / 180.0
            ax, ay = math.cos(a), math.sin(a)
            if self.blocked_at(x + ax * 11.0, y + ay * 11.0):
                continue
            if self.blocked_at(x + ax * 6.0, y + ay * 6.0):
                continue
            align = ax * ux + ay * uy
            if best is None or align > best[0]:
                best = (align, ax, ay)
        if best is None:
            return mx, my
        return best[1] * mag, best[2] * mag

    def path(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        threats: Optional[Sequence[Tuple[float, float]]] = None,
        max_expansions: int = 1600,
    ) -> List[Vec]:
        """Coarse A* (16 m) with a soft penalty near known enemies, then string-pull."""
        if not self.ok:
            return [(x1, y1)]
        x1, y1 = self.nearest_open(x1, y1)
        si, sj = self._coarse(x0, y0)
        gi, gj = self._coarse(x1, y1)
        if not self._coarse_open(gi, gj):
            gi, gj = self._nearest_coarse_open(gi, gj)
        if (si, sj) == (gi, gj):
            return [(x1, y1)]
        cnx = (self.nx + 1) // 2
        cny = (self.ny + 1) // 2
        penalties = []
        for tx, ty in threats or ():
            penalties.append((self._coarse(tx, ty), tx, ty))

        def heur(i: int, j: int) -> float:
            return math.hypot(i - gi, j - gj)

        def extra(i: int, j: int) -> float:
            cost = 0.0
            for (ci, cj), tx, ty in penalties:
                d = math.hypot(i - ci, j - cj)
                if d < 5:
                    cost += (5 - d) * 1.6
            return cost

        heap = [(heur(si, sj), 0.0, si, sj)]
        came: Dict[Tuple[int, int], Tuple[int, int]] = {}
        gscore = {(si, sj): 0.0}
        found = False
        seen = 0
        while heap and seen < max_expansions:
            _, g, i, j = heapq.heappop(heap)
            if g > gscore.get((i, j), 1e18) + 1e-9:
                continue
            if (i, j) == (gi, gj):
                found = True
                break
            seen += 1
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if ni < 0 or nj < 0 or ni >= cnx or nj >= cny:
                        continue
                    if not self._coarse_open(ni, nj):
                        continue
                    if di != 0 and dj != 0:
                        if not self._coarse_open(i + di, j) or not self._coarse_open(i, j + dj):
                            continue
                    step = 1.41421356 if di and dj else 1.0
                    ng = g + step + extra(ni, nj)
                    if ng < gscore.get((ni, nj), 1e18):
                        gscore[(ni, nj)] = ng
                        came[(ni, nj)] = (i, j)
                        heapq.heappush(heap, (ng + heur(ni, nj), ng, ni, nj))
        if not found:
            if not gscore:
                return [(x1, y1)]
            gi, gj = min(gscore, key=lambda c: (c[0] - gi) ** 2 + (c[1] - gj) ** 2)
        cells = [(gi, gj)]
        cur = (gi, gj)
        guard = 0
        while cur != (si, sj) and cur in came and guard < 10000:
            cur = came[cur]
            cells.append(cur)
            guard += 1
        cells.reverse()
        pts = [self._coarse_center(i, j) for i, j in cells]
        pts.append((x1, y1))
        return self._string_pull(x0, y0, pts)

    def _coarse(self, x: float, y: float) -> Tuple[int, int]:
        i, j = self.cell_at(x, y)
        return i // 2, j // 2

    def _coarse_open(self, i: int, j: int) -> bool:
        # Open when at least one of the 2x2 fine cells can be stood in.
        for dj in (0, 1):
            for di in (0, 1):
                if not self.is_blocked_cell(i * 2 + di, j * 2 + dj):
                    return True
        return False

    def _nearest_coarse_open(self, i: int, j: int) -> Tuple[int, int]:
        if self._coarse_open(i, j):
            return i, j
        for r in range(1, 8):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    if self._coarse_open(i + di, j + dj):
                        return i + di, j + dj
        return i, j

    def _coarse_center(self, i: int, j: int) -> Vec:
        # Prefer an open fine cell inside the coarse cell.
        for dj in (0, 1):
            for di in (0, 1):
                fi, fj = i * 2 + di, j * 2 + dj
                if not self.is_blocked_cell(fi, fj):
                    return self.minx + (fi + 0.5) * self.cell, self.miny + (fj + 0.5) * self.cell
        return self.minx + (i * 2 + 1) * self.cell, self.miny + (j * 2 + 1) * self.cell

    def _string_pull(self, x: float, y: float, pts: List[Vec]) -> List[Vec]:
        if not pts:
            return []
        out: List[Vec] = []
        ax, ay = x, y
        idx = 0
        while idx < len(pts):
            furthest = idx
            for k in range(idx, len(pts)):
                if self.los(ax, ay, pts[k][0], pts[k][1]):
                    furthest = k
                else:
                    break
            if furthest == idx and not self.los(ax, ay, pts[idx][0], pts[idx][1]):
                # Corner: still step to the next grid point so we follow the wall.
                out.append(pts[idx])
                ax, ay = pts[idx]
                idx += 1
                continue
            out.append(pts[furthest])
            ax, ay = pts[furthest]
            if furthest >= len(pts) - 1:
                break
            idx = furthest if furthest > idx else idx + 1
        return out
