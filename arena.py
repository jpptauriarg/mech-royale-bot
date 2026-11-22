"""Static map model for the bot: a 2 m height grid built from the arena files.

It answers three questions fast enough for a 50 ms decision in pure Python:
- line of sight / line of fire between two points (buildings and terrain block it),
- whether a mech can walk straight from one point to another,
- a walking path to a goal (A* that can be spread over several decisions).

Heights are stored as int16 centimeters. A cell's surface is the terrain, raised to the
roof of any building or stadium piece on it, or to a bridge deck you walk on. Elevated
bridges with head clearance are left out: mechs walk and shoot under them.
"""
import heapq
import json
import math
import os
import sys
import time
from array import array

SQRT2 = math.sqrt(2.0)
# 8 neighbours: (di, dj, step length in cells)
NEIGHBOURS = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
)


def _find(files, suffix):
    for f in files or []:
        if str(f).endswith(suffix):
            return f
    return None


class Arena:
    def __init__(self, map_dir, files=None, cell=2.0, step_up=1.2, mech_height=9.0):
        files = files or ["v2/world.json", "v2/terrain.i16"]
        wpath = os.path.join(map_dir, _find(files, "world.json") or "v2/world.json")
        with open(wpath) as f:
            world = json.load(f)
        t = world["terrain"]
        tpath = os.path.join(os.path.dirname(wpath), t.get("file", "terrain.i16"))
        tnx, tny = int(t["nx"]), int(t["ny"])
        tstep = float(t.get("step", 1.0))
        scale_cm = float(t.get("scale", 0.01)) * 100.0
        ter = array("h")
        with open(tpath, "rb") as f:
            ter.frombytes(f.read())
        if sys.byteorder == "big":
            ter.byteswap()

        self.C = C = float(cell)
        self.invC = 1.0 / C
        self.x0 = float(t["minX"])
        self.y0 = float(t["minY"])
        span_x = (tnx - 1) * tstep
        span_y = (tny - 1) * tstep
        self.W = W = max(2, int(span_x / C))
        self.H = H = max(2, int(span_y / C))
        self.max_x = self.x0 + W * C - 0.01
        self.max_y = self.y0 + H * C - 0.01
        b = world.get("bounds") or {}
        self.bx0 = float(b.get("minX", self.x0)) + 2.6
        self.by0 = float(b.get("minY", self.y0)) + 2.6
        self.bx1 = min(float(b.get("maxX", self.max_x)), self.max_x) - 2.6
        self.by1 = min(float(b.get("maxY", self.max_y)), self.max_y) - 2.6

        # Ground: the terrain vertex nearest each cell centre.
        ground = array("h", bytes(2 * W * H))
        off = C * 0.5 / tstep
        for j in range(H):
            tj = min(tny - 1, int(round((j * C) / tstep + off)))
            base = tj * tnx
            for i in range(W):
                ti = min(tnx - 1, int(round((i * C) / tstep + off)))
                v = ter[base + ti]
                ground[j * W + i] = v if scale_cm == 1.0 else int(v * scale_cm)
        self.ground = ground
        surf = array("h", ground)
        bld = bytearray(W * H)  # 1 = a building / stadium / low bridge raised this cell
        head_cm = mech_height * 100.0

        for p in world.get("prisms") or []:
            kind = p.get("kind", "building")
            base_cm = float(p.get("base", 0.0)) * 100.0
            top_cm = int(float(p.get("top", 0.0)) * 100.0)
            polys = [p.get("ring") or []] + list(p.get("holes") or [])
            edges = []
            for poly in polys:
                n = len(poly) // 2
                for k in range(n):
                    xa, ya = poly[2 * k], poly[2 * k + 1]
                    xb, yb = poly[2 * ((k + 1) % n)], poly[2 * ((k + 1) % n) + 1]
                    if ya != yb:
                        edges.append((xa, ya, xb, yb))
            ring = polys[0]
            if len(ring) < 6:
                continue
            ys = ring[1::2]
            j0 = max(0, int((min(ys) - self.y0) * self.invC) - 1)
            j1 = min(H - 1, int((max(ys) - self.y0) * self.invC) + 1)
            cells = []
            for j in range(j0, j1 + 1):
                yc = self.y0 + (j + 0.5) * C
                xs = []
                for xa, ya, xb, yb in edges:
                    if (ya <= yc) != (yb <= yc):
                        xs.append(xa + (yc - ya) * (xb - xa) / (yb - ya))
                xs.sort()
                for k in range(0, len(xs) - 1, 2):
                    ia = max(0, int(math.ceil((xs[k] - self.x0) * self.invC - 0.5)))
                    ib = min(W - 1, int(math.floor((xs[k + 1] - self.x0) * self.invC - 0.5)))
                    for i in range(ia, ib + 1):
                        cells.append(j * W + i)
            # Thin pieces can miss every cell centre: also stamp the outline.
            cells.extend(self._outline(ring))
            for idx in cells:
                g = ground[idx]
                if kind == "bridge":
                    if base_cm - g >= head_cm:
                        continue  # walk and shoot under it
                if top_cm > surf[idx]:
                    surf[idx] = top_cm
                if top_cm - g > 120:
                    bld[idx] = 1
        self.surf = surf
        self.bld = bld
        self._walls_and_costs()
        self._encl = {}

    # ------------------------------------------------------------------ build helpers
    def _outline(self, ring):
        n = len(ring) // 2
        out = []
        for k in range(n):
            xa, ya = ring[2 * k], ring[2 * k + 1]
            xb, yb = ring[2 * ((k + 1) % n)], ring[2 * ((k + 1) % n) + 1]
            seg = math.hypot(xb - xa, yb - ya)
            steps = max(1, int(seg / (self.C * 0.5)))
            for s in range(steps + 1):
                f = s / steps
                i = int((xa + (xb - xa) * f - self.x0) * self.invC)
                j = int((ya + (yb - ya) * f - self.y0) * self.invC)
                if 0 <= i < self.W and 0 <= j < self.H:
                    out.append(j * self.W + i)
        return out

    def _walls_and_costs(self):
        """wall[c] = 1 when a 4-neighbour rises more than a walkable step above c.
        cost[c] = extra planning cost near walls (keeps paths a mech-width off them)."""
        W, H, surf = self.W, self.H, self.surf
        wall = bytearray(W * H)
        thr = 150
        for j in range(H):
            r0 = j * W
            row = surf[r0:r0 + W]
            for i in range(W - 1):
                a = row[i]
                b = row[i + 1]
                if b - a > thr:
                    wall[r0 + i] = 1
                elif a - b > thr:
                    wall[r0 + i + 1] = 1
        for j in range(H - 1):
            r0 = j * W
            r1 = r0 + W
            up = surf[r1:r1 + W]
            row = surf[r0:r0 + W]
            for i in range(W):
                a = row[i]
                b = up[i]
                if b - a > thr:
                    wall[r0 + i] = 1
                elif a - b > thr:
                    wall[r1 + i] = 1
        # The map edge is a wall too.
        for i in range(W):
            wall[i] = 1
            wall[(H - 1) * W + i] = 1
        for j in range(H):
            wall[j * W] = 1
            wall[j * W + W - 1] = 1
        self.wall = wall
        # Dilate with row bitmasks: near1 = within 1 cell of a wall cell, near2 = within 2.
        rows = []
        for j in range(H):
            s = wall[j * W:(j + 1) * W]
            rows.append(_row_mask(s))
        def dilate(rs):
            out = []
            full = (1 << W) - 1
            for j in range(H):
                m = rs[j]
                if j > 0:
                    m |= rs[j - 1]
                if j < H - 1:
                    m |= rs[j + 1]
                m = (m | (m << 1) | (m >> 1)) & full
                out.append(m)
            return out
        n1 = dilate(rows)
        n2 = dilate(n1)
        cost = bytearray(W * H)
        for j in range(H):
            a = _mask_row(n1[j], W)
            b = _mask_row(n2[j], W)
            r0 = j * W
            for i in range(W):
                if a[i]:
                    cost[r0 + i] = 5
                elif b[i]:
                    cost[r0 + i] = 1
        self.cost = cost

    # ------------------------------------------------------------------ queries
    def cell(self, x, y):
        i = int((x - self.x0) * self.invC)
        j = int((y - self.y0) * self.invC)
        if i < 0:
            i = 0
        elif i >= self.W:
            i = self.W - 1
        if j < 0:
            j = 0
        elif j >= self.H:
            j = self.H - 1
        return i, j

    def center(self, i, j):
        return self.x0 + (i + 0.5) * self.C, self.y0 + (j + 0.5) * self.C

    def surface(self, x, y):
        i, j = self.cell(x, y)
        return self.surf[j * self.W + i] * 0.01

    def ground_z(self, x, y):
        i, j = self.cell(x, y)
        return self.ground[j * self.W + i] * 0.01

    def clamp(self, x, y):
        return (min(self.bx1, max(self.bx0, x)), min(self.by1, max(self.by0, y)))

    def los(self, ax, ay, az, bx, by, bz, spacing=1.0, margin=0.0):
        """True when the straight segment a->b clears every building and the terrain.
        With a margin (m), an obstacle only counts if it rises that far above the line: use it
        when "hidden" must be sure (grazing lines over terrain are what a coarse grid gets wrong)."""
        dx = bx - ax
        dy = by - ay
        d = math.hypot(dx, dy)
        n = int(d / spacing)
        if n <= 1:
            return True
        inv = 1.0 / n
        invC = self.invC
        sx = dx * inv * invC
        sy = dy * inv * invC
        sz = (bz - az) * inv * 100.0
        x = (ax - self.x0) * invC
        y = (ay - self.y0) * invC
        z = az * 100.0 + margin * 100.0
        W = self.W
        surf = self.surf
        lim = len(surf)
        for _ in range(n - 1):
            x += sx
            y += sy
            z += sz
            k = int(y) * W + int(x)
            if 0 <= k < lim and surf[k] > z:
                return False
        return True

    def ray_free(self, ax, ay, az, yaw, max_d, spacing=1.0):
        """Distance a horizontal ray at height az travels before a building or the terrain stops it."""
        cx = math.cos(yaw) * spacing
        cy = math.sin(yaw) * spacing
        invC = self.invC
        x = (ax - self.x0) * invC
        y = (ay - self.y0) * invC
        sx = cx * invC
        sy = cy * invC
        z = az * 100.0
        W, H = self.W, self.H
        surf = self.surf
        n = int(max_d / spacing)
        for k in range(1, n + 1):
            x += sx
            y += sy
            ix = int(x)
            iy = int(y)
            if ix < 0 or iy < 0 or ix >= W or iy >= H or surf[iy * W + ix] > z:
                return k * spacing
        return max_d

    def walkable(self, ax, ay, bx, by, strict=True):
        """True when a mech can walk the straight segment: no rise above a step, and (strict)
        not grazing a wall cell."""
        dx = bx - ax
        dy = by - ay
        d = math.hypot(dx, dy)
        n = int(d / (self.C * 0.5)) + 1
        inv = 1.0 / n
        invC = self.invC
        x = (ax - self.x0) * invC
        y = (ay - self.y0) * invC
        sx = dx * inv * invC
        sy = dy * inv * invC
        W, H = self.W, self.H
        surf, wall, cost = self.surf, self.wall, self.cost
        prev = surf[self._idx(ax, ay)]
        start_cell = self._idx(ax, ay)
        for _ in range(n):
            x += sx
            y += sy
            ix = int(x)
            iy = int(y)
            if ix < 1 or iy < 1 or ix >= W - 1 or iy >= H - 1:
                return False
            k = iy * W + ix
            h = surf[k]
            if h - prev > 150:
                return False
            prev = h
            if strict and k != start_cell and cost[k] >= 5:
                return False
        return True

    def _idx(self, x, y):
        i, j = self.cell(x, y)
        return j * self.W + i

    def free_near(self, x, y, max_r=12):
        """Nearest cell centre to (x, y) that is not next to a wall (spiral search)."""
        i0, j0 = self.cell(x, y)
        W, H, cost = self.W, self.H, self.cost
        if cost[j0 * W + i0] < 5:
            return self.center(i0, j0)
        for r in range(1, max_r + 1):
            best = None
            bd = 1e9
            for dj in range(-r, r + 1):
                for di in (-r, r) if abs(dj) != r else range(-r, r + 1):
                    i, j = i0 + di, j0 + dj
                    if 0 < i < W - 1 and 0 < j < H - 1 and cost[j * W + i] < 5:
                        d = di * di + dj * dj
                        if d < bd:
                            bd = d
                            best = (i, j)
            if best:
                return self.center(*best)
        return x, y

    def enclosure(self, x, y, z):
        """Cover quality of a spot: (blocked fraction of 16 directions within 35 m, mean free distance).
        Cached per cell."""
        i, j = self.cell(x, y)
        key = j * self.W + i
        hit = self._encl.get(key)
        if hit is not None:
            return hit
        cx, cy = self.center(i, j)
        zb = z + 5.0
        blocked = 0
        total = 0.0
        for k in range(16):
            d = self.ray_free(cx, cy, zb, k * (math.pi / 8.0), 60.0, 1.5)
            total += d
            if d < 35.0:
                blocked += 1
        res = (blocked / 16.0, total / 16.0)
        self._encl[key] = res
        return res


def _row_mask(row):
    m = 0
    for i, v in enumerate(row):
        if v:
            m |= 1 << i
    return m


def _mask_row(m, W):
    s = bin(m)[2:].zfill(W)[::-1]
    return [c == "1" for c in s[:W]]


class PathJob:
    """A* from start to goal over the grid, resumable: call step(max_expansions) until done.
    Moves may go down any drop but up at most one step; wall-hugging cells cost extra."""

    def __init__(self, arena, sx, sy, gx, gy, blocked=None, max_nodes=60000, weight=1.0,
                 avoid=None, avoid_r=60.0, avoid_cost=3.0):
        self.a = arena
        si, sj = arena.cell(sx, sy)
        gi, gj = arena.cell(gx, gy)
        self.start = sj * arena.W + si
        self.goal = gj * arena.W + gi
        self.gi, self.gj = gi, gj
        self.blocked = blocked or set()
        self.g = {self.start: 0.0}
        self.parent = {self.start: -1}
        self.heap = [(self._h(si, sj), 0.0, self.start)]
        self.done = False
        self.path = None
        self.best = (self._h(si, sj), self.start)
        self.expanded = 0
        self.max_nodes = max_nodes
        self.w = float(weight)
        # Cells near known enemies cost extra: avoid = [(x, y), ...] in world metres.
        self.avoid = [((ax - arena.x0) * arena.invC, (ay - arena.y0) * arena.invC) for ax, ay in (avoid or [])][:6]
        self.avoid_r2 = (avoid_r * arena.invC) ** 2
        self.avoid_cost = float(avoid_cost)
        self.heap = [(self._h(si, sj) * self.w, 0.0, self.start)]

    def _h(self, i, j):
        dx = abs(i - self.gi)
        dy = abs(j - self.gj)
        return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)

    def step(self, budget, deadline=None):
        """Expand up to `budget` nodes, or until time.perf_counter() passes `deadline`."""
        if self.done:
            return True
        a = self.a
        W, H = a.W, a.H
        surf, cost, bld = a.surf, a.cost, a.bld
        g, parent, heap = self.g, self.parent, self.heap
        goal = self.goal
        blocked = self.blocked
        gi, gj = self.gi, self.gj
        wh = self.w
        avoid = self.avoid
        ar2 = self.avoid_r2
        ac = self.avoid_cost
        n = 0
        clock = time.perf_counter
        while heap and n < budget:
            if deadline is not None and (n & 127) == 127 and clock() > deadline:
                break
            f, gc, cur = heapq.heappop(heap)
            if gc > g.get(cur, 1e18) + 1e-9:
                continue
            n += 1
            ci = cur % W
            cj = cur // W
            hcur = (abs(ci - gi) + abs(cj - gj))
            if hcur < self.best[0]:
                self.best = (hcur, cur)
            if cur == goal:
                self._finish(cur)
                return True
            hs = surf[cur]
            for di, dj, ln in NEIGHBOURS:
                ni = ci + di
                nj = cj + dj
                if ni < 1 or nj < 1 or ni >= W - 1 or nj >= H - 1:
                    continue
                nb = nj * W + ni
                rise = surf[nb] - hs
                lim = 150 if (bld[nb] or bld[cur]) else 230
                if di and dj:
                    lim = lim * 1.3
                    # no corner cutting past a wall
                    if surf[cj * W + ni] - hs > 150 or surf[nj * W + ci] - hs > 150:
                        continue
                if rise > lim or nb in blocked:
                    continue
                extra = cost[nb]
                if avoid:
                    for ax_, ay_ in avoid:
                        ddx = ni - ax_
                        ddy = nj - ay_
                        if ddx * ddx + ddy * ddy < ar2:
                            extra += ac
                ng = gc + ln * (1.0 + extra)
                if ng < g.get(nb, 1e18):
                    g[nb] = ng
                    parent[nb] = cur
                    dx = abs(ni - gi)
                    dy = abs(nj - gj)
                    heapq.heappush(heap, (ng + ((dx + dy) + (SQRT2 - 2.0) * min(dx, dy)) * wh, ng, nb))
        self.expanded += n
        if not heap or self.expanded > self.max_nodes:
            # Unreachable (or too far): go to the closest cell we reached.
            self._finish(self.best[1])
            return True
        return False

    def _finish(self, cur):
        a = self.a
        cells = []
        while cur != -1:
            cells.append(cur)
            cur = self.parent.get(cur, -1)
        cells.reverse()
        self.path = [a.center(c % a.W, c // a.W) for c in cells]
        self.done = True
