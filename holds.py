"""Jettable roofs a ground walker cannot path onto.

Building holes in this map are not open courtyards: a second prism fills almost
every hole, so there is no floor inside them. What is reachable, and what house
bots that stay on the ground cannot walk onto, is the roof. 100 energy climbs
about 60 m. This list keeps roofs from 14 m to 48 m: tall enough that a nearby
ground eye hits the wall, short enough to jet with energy left. A far eye still
clears the edge (pitch does not matter), so a roof is cover from the bots at
the wall, not a cloak from the whole circle.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import nav

Vec = Tuple[float, float]
_CACHE: Dict[str, List["Hold"]] = {}


class Hold:
    def __init__(self, x: float, y: float, z: float, wall: float, ring: Tuple[float, ...], holes: Tuple[Tuple[float, ...], ...]):
        self.x = x
        self.y = y
        self.z = z
        self.wall = wall
        self.ring = ring
        self.holes = holes
        xs = ring[0::2]
        ys = ring[1::2]
        self.minx = min(xs)
        self.maxx = max(xs)
        self.miny = min(ys)
        self.maxy = max(ys)

    def contains(self, x: float, y: float) -> bool:
        if x < self.minx or x > self.maxx or y < self.miny or y > self.maxy:
            return False
        if not nav._pip(x, y, self.ring):
            return False
        return not self.in_hole(x, y)

    def in_hole(self, x: float, y: float) -> bool:
        return any(nav._pip(x, y, hole) for hole in self.holes)

    def gap(self, x: float, y: float) -> float:
        """Distance to the footprint. Zero when this point is on the roof."""
        if self.contains(x, y):
            return 0.0
        return _edge_distance(x, y, self.ring)


def _area(ring: Sequence[float]) -> float:
    n = len(ring) // 2
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += ring[2 * i] * ring[2 * j + 1] - ring[2 * j] * ring[2 * i + 1]
    return abs(a) * 0.5


def _interior(ring: Sequence[float]) -> Optional[Vec]:
    n = len(ring) // 2
    acc = cx = cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = ring[2 * i] * ring[2 * j + 1] - ring[2 * j] * ring[2 * i + 1]
        acc += cross
        cx += (ring[2 * i] + ring[2 * j]) * cross
        cy += (ring[2 * i + 1] + ring[2 * j + 1]) * cross
    if abs(acc) > 1e-6:
        acc *= 0.5
        px, py = cx / (6 * acc), cy / (6 * acc)
        if nav._pip(px, py, ring):
            return px, py
    xs = ring[0::2]
    ys = ring[1::2]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    best = None
    best_d = 1e18
    mid_x = (minx + maxx) * 0.5
    mid_y = (miny + maxy) * 0.5
    y = miny + 2.0
    while y < maxy:
        x = minx + 2.0
        while x < maxx:
            if nav._pip(x, y, ring):
                d = (x - mid_x) ** 2 + (y - mid_y) ** 2
                if d < best_d:
                    best_d = d
                    best = (x, y)
            x += 4.0
        y += 4.0
    return best


def _edge_distance(x: float, y: float, ring: Sequence[float]) -> float:
    n = len(ring) // 2
    best = 1e9
    for i in range(n):
        j = (i + 1) % n
        x1, y1 = ring[2 * i], ring[2 * i + 1]
        x2, y2 = ring[2 * j], ring[2 * j + 1]
        dx, dy = x2 - x1, y2 - y1
        length2 = dx * dx + dy * dy or 1.0
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length2))
        best = min(best, math.hypot(x - (x1 + dx * t), y - (y1 + dy * t)))
    return best


def _extract(arena: nav.Arena, map_dir: str) -> List[Hold]:
    import json

    path = os.path.join(map_dir, "v2", "world.json")
    with open(path) as f:
        world = json.load(f)
    holds: List[Hold] = []
    for prism in world.get("prisms") or []:
        if prism.get("kind") != "building":
            continue
        ring = prism.get("ring") or []
        if _area(ring) < 180.0:
            continue
        holes = tuple(tuple(float(v) for v in hole) for hole in (prism.get("holes") or []))
        point = _interior(ring)
        if point is None:
            continue
        x, y = point
        if any(nav._pip(x, y, hole) for hole in holes):
            continue
        if _edge_distance(x, y, ring) < 4.5:
            continue
        floor = arena.height(x, y)
        top = float(prism["top"])
        wall = top - floor
        # Below 14 m a ground eye at the wall sees over the edge. Above 48 m
        # the climb spends the whole tank and a late jump fails.
        if wall < 14.0 or wall > 48.0:
            continue
        holds.append(Hold(x, y, top, wall, tuple(float(v) for v in ring), holes))
    holds.sort(key=lambda hold: -hold.wall)
    return holds


def get_holds(map_dir: str) -> List[Hold]:
    key = os.path.abspath(map_dir)
    found = _CACHE.get(key)
    if found is None:
        arena = nav.get_arena(key)
        found = _extract(arena, key) if arena.ok else []
        _CACHE[key] = found
    return found


def hold_at(holds: Sequence[Hold], x: float, y: float) -> Optional[Hold]:
    for hold in holds:
        if hold.contains(x, y):
            return hold
    return None
