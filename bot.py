"""Shade: a survival-first mechfight/1 bot.

Placement beats kills. Stay inside the closing circle, keep buildings between
this mech and other mechs, walk quietly when someone is close, and only shoot
when a fight is already happening or the circle is too small to hide in.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import mechfight as mf
import nav

Vec = Tuple[float, float]

# Higher is better when an upgrade offer arrives. Survival stats come first.
UPGRADE_RANK = {
    "armor": 100,
    "plating": 96,
    "nanites": 92,
    "servos": 84,
    "jets": 78,
    "sensors": 72,
    "capacitor": 64,
    "reactor": 60,
    "gyros": 54,
    "gun_mag": 48,
    "gun_velocity": 44,
    "gun_damage": 40,
    "gun_rate": 38,
    "lightning_efficiency": 36,
    "lightning_range": 34,
    "mortar_velocity": 30,
    "mortar_reload": 28,
    "laser_charge": 22,
    "mortar_splash": 20,
    "lightning_chain": 18,
    "laser_damage": 14,
    "laser_pierce": 12,
}
LOOT_RANK = {"lightning": 30, "mortar": 18, "laser": 10, "gun": 0}
SOUND_DIST = {"near": 40.0, "mid": 110.0, "far": 210.0}
LOUD = {"gun", "mortar", "explosion", "laser", "lightning", "laser_charge", "boost", "jet", "step"}


def ang_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _slot(me: Dict[str, Any], i: int) -> Optional[Dict[str, Any]]:
    weapons = me.get("weapons") or []
    if i >= len(weapons):
        return None
    return weapons[i]


class Bot:
    name = "shade"

    def __init__(self, hello: mf.Hello):
        cfg = hello["config"]
        self.slot = int(hello["slot"])
        self.arena = nav.get_arena(hello["map"]["dir"])
        mech = cfg["mech"]
        self.walk = float(mech.get("walk_speed", 11.7))
        self.reach = float(cfg["loot"]["pickup_radius"])
        self.cube_r = float(cfg["cubes"]["pickup_radius"])
        self.gun_speed = float(cfg["weapons"]["gun"].get("speed", 252))
        self.gun_range = float(cfg["weapons"]["gun"].get("range", 230))
        self.mortar_speed = float(cfg["weapons"]["mortar"].get("speed", 72))
        self.mortar_g = float(cfg["weapons"]["mortar"].get("gravity", 22))
        step = float(cfg["sound"].get("step_speed", 5))
        # Move slower than step_speed so nearby mechs do not hear footsteps.
        self.quiet_scale = clamp((step - 0.7) / self.walk, 0.22, 0.45)
        self.mem: Dict[int, Dict[str, Any]] = {}
        self.npcs: Dict[int, Dict[str, Any]] = {}
        self.phantoms: List[Dict[str, Any]] = []
        self.waypoints: List[Vec] = []
        self.wp_i = 0
        self.goal: Optional[Vec] = None
        self.plan_t = -1e9
        self.trail: List[Tuple[float, float, float]] = []
        self.unstick_until = -1.0
        self.unstick_dir: Vec = (1.0, 0.0)
        self.hurt_until = -1.0
        self.hurt_bearing: Optional[float] = None
        self.hurt_started = -1.0
        self.last_hit_t = -1e9
        self.last_hit_by: Optional[float] = None
        self.was_out = False

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = (float(v) for v in me["pos"])
        t = float(obs["t"])
        self._ingest(obs, x, y, t)
        threats = self._threats(x, y, t)
        npcs = self._npc_list(x, y, t)
        urgent = self._needs_next(obs, x, y)
        self._replan(obs, x, y, threats, urgent)
        target, fight = self._pick_target(obs, threats, x, y, z)
        dodge_x, dodge_y, dodge_jump = self._dodge(obs, x, y, z)
        move, boost, jump = self._steer(obs, x, y, threats, npcs, dodge_x, dodge_y, fight, target, urgent)
        jump = jump or dodge_jump or self._stuck(obs, x, y, move)
        if self.unstick_until > t:
            move = [self.unstick_dir[0], self.unstick_dir[1]]
            jump = True
            if me["energy"] > 16 and not me["in_zone"]:
                boost = True
        aim, fire = self._aim_fire(obs, me, x, y, z, target, fight, move)
        action: mf.Action = {
            "tick": obs["tick"],
            "move": move,
            "boost": bool(boost),
            "jump": bool(jump and me.get("grounded", True)),
            "aim": aim,
            "fire": fire,
        }
        taken = self._take(obs, me, x, y, z, threats)
        if taken is not None:
            action["take"] = taken[0]
            if taken[1] is not None:
                action["drop"] = taken[1]
        if self._should_reload(me, fight):
            action["reload"] = True
        offer = obs.get("offer")
        if offer:
            action["pick"] = self._pick_upgrade(offer)
        mf.draw.label(self._label(me, threats, fight, urgent))
        return action

    # ---------------------------------------------------------------- perception

    def _ingest(self, obs: mf.Observation, x: float, y: float, t: float) -> None:
        seen = set()
        for m in obs.get("mechs") or []:
            mid = int(m["id"])
            pos = m["pos"]
            vel = m.get("vel") or [0, 0, 0]
            rec = {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "vx": float(vel[0]),
                "vy": float(vel[1]),
                "hp": float(m.get("hp") or 1400),
                "t": t,
                "weapons": list(m.get("weapons") or []),
                "charging": bool(m.get("charging")),
                "aim": float((m.get("aim") or [0, 0])[0]),
                "fresh": True,
            }
            if m.get("kind") == "npc":
                self.npcs[mid] = rec
            else:
                self.mem[mid] = rec
                seen.add(mid)
        for mid, rec in list(self.mem.items()):
            if mid not in seen:
                rec["fresh"] = False
            if t - rec["t"] > 8.0:
                del self.mem[mid]
        for mid, rec in list(self.npcs.items()):
            if t - rec["t"] > 2.0:
                del self.npcs[mid]
        for s in obs.get("sounds") or []:
            if s.get("kind") not in LOUD:
                continue
            band = s.get("dist", "mid")
            dist = SOUND_DIST.get(band, 110.0)
            bearing = float(s["bearing"])
            px = x + math.cos(bearing) * dist
            py = y + math.sin(bearing) * dist
            if any(math.hypot(px - m["x"], py - m["y"]) < 35 for m in self.mem.values()):
                continue
            self.phantoms.append({"x": px, "y": py, "t": t, "near": band == "near"})
        if len(self.phantoms) > 10:
            self.phantoms = self.phantoms[-10:]
        self.phantoms = [p for p in self.phantoms if t - p["t"] < 3.0]
        for e in obs.get("events") or []:
            if e.get("type") != "damaged":
                continue
            # Zone damage has no attacker. Hiding from it means walking inward, not shooting.
            if e.get("weapon") == "zone" or e.get("by") is None or e.get("bearing") is None:
                continue
            self.last_hit_t = t
            self.last_hit_by = e.get("by")
            self.hurt_bearing = float(e["bearing"])
            self.hurt_until = t + 3.5
            self.hurt_started = t

    def _threats(self, x: float, y: float, t: float) -> List[Dict[str, Any]]:
        out = []
        for mid, m in self.mem.items():
            age = min(1.2, max(0.0, t - m["t"]))
            px = m["x"] + m["vx"] * age
            py = m["y"] + m["vy"] * age
            out.append(
                {
                    "id": mid,
                    "x": px,
                    "y": py,
                    "z": m["z"],
                    "vx": m["vx"],
                    "vy": m["vy"],
                    "hp": m["hp"],
                    "dist": math.hypot(px - x, py - y),
                    "fresh": m["fresh"],
                    "charging": m["charging"],
                    "phantom": False,
                    "weapons": m["weapons"],
                }
            )
        for p in self.phantoms:
            out.append(
                {
                    "id": -1,
                    "x": p["x"],
                    "y": p["y"],
                    "z": 0.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "hp": 1400.0,
                    "dist": math.hypot(p["x"] - x, p["y"] - y),
                    "fresh": False,
                    "charging": False,
                    "phantom": True,
                    "near": p["near"],
                    "weapons": [],
                }
            )
        return out

    def _npc_list(self, x: float, y: float, t: float) -> List[Dict[str, Any]]:
        out = []
        for m in self.npcs.values():
            out.append({"x": m["x"], "y": m["y"], "dist": math.hypot(m["x"] - x, m["y"] - y)})
        return out

    # ---------------------------------------------------------------- decisions

    def _needs_next(self, obs: mf.Observation, x: float, y: float) -> bool:
        zone = obs["zone"]
        t = float(obs["t"])
        me = obs["self"]
        if not me["in_zone"]:
            return True
        next_c = zone["next_center"]
        next_r = float(zone["next_radius"])
        margin = self._margin(next_r)
        d_next = math.hypot(x - next_c[0], y - next_c[1])
        outside = d_next > max(0.0, next_r - margin)
        travel = max(0.0, d_next - max(0.0, next_r - margin)) / 11.0
        if t >= float(zone["shrink_start"]):
            return True
        if outside and travel + 10.0 > max(0.0, float(zone["shrink_start"]) - t):
            return True
        if float(zone["shrink_end"]) - t < travel + 8.0:
            return True
        return False

    def _dest(self, obs: mf.Observation, urgent: bool) -> Tuple[Vec, float]:
        zone = obs["zone"]
        if urgent:
            return (float(zone["next_center"][0]), float(zone["next_center"][1])), float(zone["next_radius"])
        return (float(zone["center"][0]), float(zone["center"][1])), float(zone["radius"])

    def _margin(self, radius: float) -> float:
        if radius > 200:
            return 32.0
        if radius > 100:
            return 16.0
        if radius > 45:
            return 8.0
        if radius > 18:
            return 3.0
        return 0.0

    def _replan(self, obs: mf.Observation, x: float, y: float, threats: List[Dict[str, Any]], urgent: bool) -> None:
        t = float(obs["t"])
        hurt_now = self.hurt_until > t and t - self.hurt_started < 0.25
        outside = not obs["self"]["in_zone"]
        out_now = outside and not self.was_out
        self.was_out = outside
        interval = 0.35 if outside or urgent else 0.7
        if self.waypoints and t - self.plan_t < interval and not hurt_now and not out_now and t >= self.unstick_until:
            return
        dest_c, dest_r = self._dest(obs, urgent)
        goal = self._choose_goal(obs, x, y, threats, urgent, dest_c, dest_r)
        self.goal = goal
        near = [(th["x"], th["y"]) for th in threats if th["dist"] < 240 and not (th.get("phantom") and not th.get("near"))]
        self.waypoints = self.arena.path(x, y, goal[0], goal[1], near)
        self.wp_i = 0
        self.plan_t = t

    def _choose_goal(
        self,
        obs: mf.Observation,
        x: float,
        y: float,
        threats: List[Dict[str, Any]],
        urgent: bool,
        dest_c: Vec,
        dest_r: float,
    ) -> Vec:
        # Outside the circle we actually have to be in: take the near edge.
        # A far-side camp is only safe once we are already inside it.
        if urgent or not obs["self"]["in_zone"]:
            return self._entry_point(x, y, dest_c, dest_r)
        if self.hurt_until > float(obs["t"]) and self.hurt_bearing is not None:
            cover = self._cover_point(x, y, self.hurt_bearing)
            if cover is not None and self._inside(cover, dest_c, dest_r):
                return cover
        drop = self._drop_goal(obs, x, y, threats)
        if drop is not None:
            return drop
        loot = self._loot_goal(obs, x, y, threats)
        if loot is not None:
            return loot
        cube = self._cube_goal(obs, x, y, threats)
        if cube is not None:
            return cube
        return self._ring_point(obs, x, y, threats, dest_c, dest_r, urgent)

    def _entry_point(self, x: float, y: float, dest_c: Vec, dest_r: float) -> Vec:
        """Nearest open point that sits inside the destination circle."""
        margin = self._margin(dest_r)
        usable = max(0.0, dest_r - margin)
        dx, dy = dest_c[0] - x, dest_c[1] - y
        d = math.hypot(dx, dy) or 1.0
        ux, uy = dx / d, dy / d
        best = None
        best_d = 1e9
        # Sample the corridor from us toward the circle, plus a few side steps,
        # and keep the closest open point that is actually inside it.
        for lateral in (0.0, 18.0, -18.0, 36.0, -36.0):
            sx, sy = -uy * lateral, ux * lateral
            for dist in (d - usable + 12.0, d * 0.55, d * 0.8, max(8.0, d - usable * 0.4), d):
                if dist < 0:
                    dist = 0.0
                px = x + ux * dist + sx
                py = y + uy * dist + sy
                if math.hypot(px - dest_c[0], py - dest_c[1]) > usable + 1.0:
                    continue
                if self.arena.blocked_at(px, py):
                    px, py = self.arena.nearest_open(px, py, 3)
                    if self.arena.blocked_at(px, py):
                        continue
                    if math.hypot(px - dest_c[0], py - dest_c[1]) > usable + 4.0:
                        continue
                travel = math.hypot(px - x, py - y)
                if travel < best_d:
                    best_d = travel
                    best = (px, py)
        if best is not None:
            return best
        return self.arena.nearest_open(dest_c[0], dest_c[1], 8)

    def _inside(self, p: Vec, c: Vec, r: float) -> bool:
        return math.hypot(p[0] - c[0], p[1] - c[1]) <= max(4.0, r - 2.0)

    def _hot(self, threats: List[Dict[str, Any]], dist: float) -> bool:
        return any((not th.get("phantom")) and th["dist"] < dist for th in threats)

    def _cover_point(self, x: float, y: float, bearing: float) -> Optional[Vec]:
        away = bearing + math.pi
        srcx = x + math.cos(bearing) * 50.0
        srcy = y + math.sin(bearing) * 50.0
        for dist in (18.0, 30.0, 44.0):
            px = x + math.cos(away) * dist
            py = y + math.sin(away) * dist
            px, py = self.arena.nearest_open(px, py, 3)
            if self.arena.blocked_at(px, py):
                continue
            if not self.arena.los(px, py, srcx, srcy):
                return px, py
        return None

    def _drop_goal(self, obs: mf.Observation, x: float, y: float, threats: List[Dict[str, Any]]) -> Optional[Vec]:
        if self._hot(threats, 110):
            return None
        me = obs["self"]
        best = None
        best_d = 1e9
        for drop in obs.get("drops") or []:
            if drop.get("holder") not in (None, me["id"]):
                continue
            if float(drop.get("lands_in") or 0) > 10:
                continue
            pos = drop["pos"]
            d = math.hypot(float(pos[0]) - x, float(pos[1]) - y)
            if d > 75 or d >= best_d:
                continue
            zone = obs["zone"]
            if math.hypot(float(pos[0]) - zone["center"][0], float(pos[1]) - zone["center"][1]) > float(zone["radius"]) - 10:
                continue
            hp_frac = me["hp"] / max(1.0, me["max_hp"])
            if hp_frac > 0.82 and me["cubes"] >= 6:
                continue
            best = (float(pos[0]), float(pos[1]))
            best_d = d
        return best

    def _holding(self, me: Dict[str, Any], kind: str) -> bool:
        for i in (0, 1):
            w = _slot(me, i)
            if w and w.get("kind") == kind:
                return True
        return False

    def _loot_goal(self, obs: mf.Observation, x: float, y: float, threats: List[Dict[str, Any]]) -> Optional[Vec]:
        if self._hot(threats, 85):
            return None
        me = obs["self"]
        if _slot(me, 1) is not None and self._holding(me, "lightning"):
            return None
        best = None
        best_s = -1e9
        for p in obs.get("pickups") or []:
            kind = p.get("kind")
            if kind not in ("lightning", "mortar", "laser"):
                continue
            if self._holding(me, kind):
                continue
            pos = p["pos"]
            d = math.hypot(float(pos[0]) - x, float(pos[1]) - y)
            if d > 36:
                continue
            score = LOOT_RANK[kind] - d * 0.45
            if score > best_s:
                best_s = score
                best = (float(pos[0]), float(pos[1]))
        return best

    def _cube_goal(self, obs: mf.Observation, x: float, y: float, threats: List[Dict[str, Any]]) -> Optional[Vec]:
        me = obs["self"]
        if me["cubes"] >= 6 or self._hot(threats, 70):
            return None
        best = None
        best_d = 1e9
        limit = 18.0 if me["cubes"] < 3 else 12.0
        for c in obs.get("cubes") or []:
            pos = c["pos"]
            d = math.hypot(float(pos[0]) - x, float(pos[1]) - y)
            if d < best_d and d <= limit and abs(float(pos[2]) - me["pos"][2]) < 4:
                best_d = d
                best = (float(pos[0]), float(pos[1]))
        return best

    def _ring_point(
        self,
        obs: mf.Observation,
        x: float,
        y: float,
        threats: List[Dict[str, Any]],
        dest_c: Vec,
        dest_r: float,
        urgent: bool,
    ) -> Vec:
        margin = self._margin(dest_r)
        usable = max(0.0, dest_r - margin)
        real = [th for th in threats if not th.get("phantom")]
        if dest_r < 55:
            stand = usable * 0.28
        elif real:
            stand = usable * 0.74
        else:
            stand = usable * 0.48
        if real:
            cx = sum(th["x"] for th in real) / len(real)
            cy = sum(th["y"] for th in real) / len(real)
            away = math.atan2(dest_c[1] - cy, dest_c[0] - cx)
        else:
            away = math.atan2(y - dest_c[1], x - dest_c[0])
        best: Vec = dest_c
        best_s = -1e18
        for delta in (0.0, 0.35, -0.35, 0.7, -0.7, 1.15, -1.15, 1.7, -1.7, 2.4, -2.4, math.pi):
            ang = away + delta
            for scale in (1.0, 0.62):
                px = dest_c[0] + math.cos(ang) * stand * scale
                py = dest_c[1] + math.sin(ang) * stand * scale
                if dest_r > 8 and math.hypot(px - dest_c[0], py - dest_c[1]) > usable + 6:
                    continue
                if self.arena.blocked_at(px, py):
                    px, py = self.arena.nearest_open(px, py, 4)
                    if self.arena.blocked_at(px, py):
                        continue
                travel = math.hypot(px - x, py - y)
                score = math.cos(delta) * 28.0 - travel * (0.25 if urgent else 0.55)
                if self.arena.has_cover(px, py):
                    score += 70.0
                for th in threats:
                    d = math.hypot(px - th["x"], py - th["y"])
                    weight = 0.45 if th.get("phantom") else 1.0
                    if d < 45:
                        score -= 420 * weight
                    elif d < 100:
                        score -= (100 - d) * 3.2 * weight
                    else:
                        score += min(d, 280) * 0.04 * weight
                    if (
                        not th.get("phantom")
                        and th.get("fresh")
                        and d < 170
                        and self.arena.los(px, py, th["x"], th["y"])
                    ):
                        score -= 140.0
                if score > best_s:
                    best_s = score
                    best = (px, py)
        if self.arena.blocked_at(best[0], best[1]):
            return self.arena.nearest_open(dest_c[0], dest_c[1], 8)
        return best

    def _waypoint(self, x: float, y: float) -> Vec:
        wps = self.waypoints or ([self.goal] if self.goal else [(x, y)])
        i = min(self.wp_i, len(wps) - 1)
        while i < len(wps) - 1 and math.hypot(wps[i][0] - x, wps[i][1] - y) < 13.0:
            i += 1
        self.wp_i = i
        return wps[i]

    def _steer(
        self,
        obs: mf.Observation,
        x: float,
        y: float,
        threats: List[Dict[str, Any]],
        npcs: List[Dict[str, Any]],
        dodge_x: float,
        dodge_y: float,
        fight: bool,
        target: Optional[Dict[str, Any]],
        urgent: bool,
    ) -> Tuple[List[float], bool, bool]:
        me = obs["self"]
        wx, wy = self._waypoint(x, y)
        dx, dy = wx - x, wy - y
        mag = math.hypot(dx, dy) or 1.0
        mx, my = dx / mag, dy / mag
        fx = fy = 0.0
        close = False
        for th in threats:
            if th["dist"] > 78:
                continue
            w = (78 - th["dist"]) / 78
            if th.get("phantom"):
                w *= 0.4
            if th["dist"] < 32 and not th.get("phantom"):
                w += 0.9
                close = True
            if th.get("charging") and th["dist"] < 80:
                w += 0.7
            d = max(th["dist"], 1.0)
            fx += (x - th["x"]) / d * w
            fy += (y - th["y"]) / d * w
        for npc in npcs:
            if npc["dist"] < 40:
                d = max(npc["dist"], 1.0)
                w = (40 - npc["dist"]) / 40
                fx += (x - npc["x"]) / d * w * 0.8
                fy += (y - npc["y"]) / d * w * 0.8
        zone = obs["zone"]
        cx, cy = float(zone["center"][0]), float(zone["center"][1])
        dcent = math.hypot(x - cx, y - cy)
        radius = float(zone["radius"])
        outside = dcent > radius - 2.0
        # Fleeing from a mech must not drag us back out of the circle, and it must
        # not override the path when the only way in goes around a building.
        if outside or urgent:
            if not close:
                fx = fy = 0.0
            else:
                fx *= 0.35
                fy *= 0.35
        if close and me["in_zone"] and not outside:
            mx, my = fx + dodge_x, fy + dodge_y
        else:
            mx = mx + fx * 0.75 + dodge_x * (0.45 if outside else 1.0)
            my = my + fy * 0.75 + dodge_y * (0.45 if outside else 1.0)
        if fight and target is not None and me["in_zone"] and not close and not outside:
            side = 1.0 if (int(obs["tick"]) // 8 + self.slot) % 2 == 0 else -1.0
            yaw = math.atan2(target["y"] - y, target["x"] - x)
            mx += -math.sin(yaw) * side * 0.85
            my += math.cos(yaw) * side * 0.85
        if not outside and not (fight and float(zone["radius"]) <= 100):
            breaker = self._break_los(x, y, mx, my, threats)
            if breaker is not None:
                mx, my = breaker
        mx, my = self.arena.avoid(x, y, mx, my)
        mag = math.hypot(mx, my)
        if mag < 1e-6:
            return [0.0, 0.0], False, False
        quiet = self._quiet(obs, threats, urgent, close)
        scale = self.quiet_scale if quiet else 1.0
        mx, my = mx / mag * scale, my / mag * scale
        boost = False
        energy = float(me["energy"])
        if not quiet and energy > 8:
            if not me["in_zone"] or outside:
                boost = True
            elif urgent and energy > 28:
                boost = True
            elif close and energy > 24:
                boost = True
        return [mx, my], boost, False

    def _break_los(self, x: float, y: float, mx: float, my: float, threats: List[Dict[str, Any]]):
        """Step toward a spot the nearest visible mech cannot see."""
        fresh = [th for th in threats if th.get("fresh") and not th.get("phantom") and th["dist"] < 120]
        if not fresh:
            return None
        th = min(fresh, key=lambda item: item["dist"])
        if not self.arena.los(x, y, th["x"], th["y"]):
            return None
        tx, ty = th["x"], th["y"]
        mag = math.hypot(mx, my) or 1.0
        ux, uy = mx / mag, my / mag
        best = None
        best_s = -1e9
        for deg in range(0, 360, 30):
            ang = deg * math.pi / 180.0
            ax, ay = math.cos(ang), math.sin(ang)
            px, py = x + ax * 16.0, y + ay * 16.0
            if self.arena.blocked_at(x + ax * 8.0, y + ay * 8.0) or self.arena.blocked_at(px, py):
                continue
            hidden = not self.arena.los(px, py, tx, ty)
            score = (90.0 if hidden else 0.0) + (ax * ux + ay * uy) * 25.0
            if hidden and self.arena.has_cover(px, py):
                score += 20.0
            if score > best_s:
                best_s = score
                best = (ax, ay)
        if best is None or best_s < 40:
            return None
        return best

    def _quiet(self, obs: mf.Observation, threats: List[Dict[str, Any]], urgent: bool, close: bool) -> bool:
        if close or urgent or not obs["self"]["in_zone"]:
            return False
        if float(obs["zone"]["radius"]) < 70:
            return False
        if any((not th.get("phantom")) and th["dist"] < 62 for th in threats):
            return True
        if any(th.get("phantom") and th.get("near") and th["dist"] < 70 for th in threats):
            return True
        return False

    def _dodge(self, obs: mf.Observation, x: float, y: float, z: float) -> Tuple[float, float, bool]:
        vx = vy = 0.0
        jump = False
        side = 1.0 if (int(obs["tick"]) // 6 + self.slot) % 2 == 0 else -1.0
        for p in obs.get("projectiles") or []:
            pos = p["pos"]
            vel = p["vel"]
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            pvx, pvy, pvz = float(vel[0]), float(vel[1]), float(vel[2])
            if pvx * (x - px) + pvy * (y - py) < 0 and math.hypot(px - x, py - y) > 12:
                continue
            g = self.mortar_g if p.get("kind") == "mortar" else 0.0
            worst = None
            for step in range(1, 8):
                tt = 0.15 * step
                qx = px + pvx * tt
                qy = py + pvy * tt
                qz = pz + pvz * tt - 0.5 * g * tt * tt
                d = math.hypot(qx - x, qy - y)
                if d < 7.5 and abs(qz - (z + 4.5)) < 7.0:
                    if worst is None or d < worst[0]:
                        worst = (d, tt, qx, qy, p.get("kind"))
            if worst is None:
                continue
            d, tt, qx, qy, kind = worst
            sp = math.hypot(pvx, pvy) or 1.0
            sx, sy = -pvy / sp * side, pvx / sp * side
            if (x - qx) * sx + (y - qy) * sy < 0:
                sx, sy = -sx, -sy
            w = (7.5 - d) / 7.5
            if tt < 0.45:
                w += 0.4
            vx += sx * w * 1.3
            vy += sy * w * 1.3
            if kind == "mortar" and tt < 0.8 and d < 8:
                jump = True
        for m in obs.get("mechs") or []:
            if m.get("kind") != "mech" or not m.get("charging"):
                continue
            pos = m["pos"]
            aim = float((m.get("aim") or [0, 0])[0])
            bearing = math.atan2(y - float(pos[1]), x - float(pos[0]))
            if abs(ang_diff(aim, bearing)) < 0.28:
                vx += -math.sin(aim) * side * 1.5
                vy += math.cos(aim) * side * 1.5
        return vx, vy, jump

    def _stuck(self, obs: mf.Observation, x: float, y: float, move: List[float]) -> bool:
        t = float(obs["t"])
        self.trail.append((t, x, y))
        self.trail = [p for p in self.trail if t - p[0] <= 1.2]
        if math.hypot(move[0], move[1]) < 0.7:
            return False
        if t < self.unstick_until:
            return True
        old = self.trail[0]
        if t - old[0] < 0.85:
            return False
        if math.hypot(x - old[1], y - old[2]) > 2.0:
            return False
        # The commanded direction is into a wall. Try the first open heading,
        # sweeping from the goal so a courtyard gap still gets found.
        gx, gy = self.goal if self.goal else (x + move[0], y + move[1])
        base = math.atan2(gy - y, gx - x)
        picked = None
        for deg in (0, 40, -40, 80, -80, 120, -120, 160, -160):
            ang = base + deg * math.pi / 180.0
            ax, ay = math.cos(ang), math.sin(ang)
            if any(self.arena.blocked_at(x + ax * dist, y + ay * dist) for dist in (6.0, 12.0, 18.0)):
                continue
            picked = (ax, ay)
            break
        if picked is None:
            ang = base + (1 if int(t) % 2 == 0 else -1) * math.pi / 2
            picked = (math.cos(ang), math.sin(ang))
        self.unstick_dir = picked
        self.unstick_until = t + 1.1
        self.plan_t = -1e9
        return True

    def _pick_target(
        self,
        obs: mf.Observation,
        threats: List[Dict[str, Any]],
        x: float,
        y: float,
        z: float,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        visible = [th for th in threats if th.get("fresh") and not th.get("phantom")]
        if not visible:
            return None, False
        target = min(visible, key=lambda th: th["dist"])
        me = obs["self"]
        zone = obs["zone"]
        n_close = sum(1 for th in visible if th["dist"] < 120)
        revealed = (
            self.last_hit_by is not None
            and self.last_hit_t > float(obs["t"]) - 1.6
            and int(self.last_hit_by) == int(target["id"])
        )
        endgame = float(zone["radius"]) <= 100 or int(obs["alive"]) <= 3
        # A second mech nearby means a squad or a third party. Leave.
        # Shoot when the circle is already small, or when exactly one mech
        # has already hit us and is still the only one in view.
        fight = False
        if n_close >= 2 and not endgame:
            fight = False
        elif endgame and n_close <= 2 and target["dist"] < 190 and me["in_zone"]:
            fight = True
        elif n_close == 1 and revealed and target["dist"] < 170 and me["in_zone"]:
            fight = True
        return target, fight

    def _aim_fire(
        self,
        obs: mf.Observation,
        me: Dict[str, Any],
        x: float,
        y: float,
        z: float,
        target: Optional[Dict[str, Any]],
        fight: bool,
        move: List[float],
    ) -> Tuple[List[float], List[bool]]:
        fire = [False, False]
        if target is not None and target["dist"] < 210:
            dist = max(1.0, target["dist"])
            tof = dist / self.gun_speed
            lx = target["x"] + target["vx"] * tof
            ly = target["y"] + target["vy"] * tof
            dist2 = math.hypot(lx - x, ly - y)
            tof = dist2 / self.gun_speed
            lx = target["x"] + target["vx"] * tof
            ly = target["y"] + target["vy"] * tof
            yaw = math.atan2(ly - y, lx - x)
            pitch = math.atan2((target["z"] + 4.5) - (z + 7.0), max(1.0, dist2))
            pitch = clamp(pitch, -0.7, 1.2)
        else:
            heard = self._glance(obs)
            if heard is not None:
                yaw = heard
            elif math.hypot(move[0], move[1]) > 0.2:
                yaw = math.atan2(move[1], move[0]) + 0.5 * math.sin(float(obs["t"]) * 1.15 + self.slot)
            else:
                yaw = float(me["aim"][0]) + 0.7 * math.sin(float(obs["t"]) * 0.8 + self.slot)
            pitch = 0.0
        if fight and target is not None:
            w1 = _slot(me, 1)
            if w1 and w1.get("kind") == "mortar" and 30 < target["dist"] < 200 and not self._hot([target], 14):
                arg = clamp(target["dist"] * self.mortar_g / (self.mortar_speed ** 2), -1.0, 1.0)
                pitch = clamp(math.pi / 2 - 0.5 * math.asin(arg), 0.2, 1.25)
            err = abs(ang_diff(float(me["aim"][0]), yaw))
            w0 = _slot(me, 0)
            if (
                w0
                and w0.get("kind") == "gun"
                and err < 0.22
                and float(w0.get("cooldown") or 0) <= 0.06
                and float(w0.get("ammo") or 0) > 0
                and float(w0.get("reload") or 0) <= 0
                and target["dist"] < self.gun_range - 5
            ):
                fire[0] = True
            if w1 and float(w1.get("cooldown") or 0) <= 0.06:
                kind = w1.get("kind")
                if kind == "lightning" and target["dist"] < float((w1.get("stats") or {}).get("range", 45)) + 2 and me["energy"] > 12:
                    if abs(ang_diff(float(me["aim"][0]), yaw)) < 0.45:
                        fire[1] = True
                elif kind == "mortar" and 30 < target["dist"] < 190 and not self._hot([target], 14):
                    fire[1] = True
                elif kind == "laser" and float(obs["zone"]["radius"]) <= 90 and target["dist"] < 220:
                    # Hold a short charge, then release. Charging is slow, so only in the endgame.
                    charge = float(w1.get("charge") or 0)
                    fire[1] = charge < 0.55
        elif target is not None and target["dist"] < 30:
            w1 = _slot(me, 1)
            if w1 and w1.get("kind") == "lightning" and me["energy"] > 8:
                fire[1] = True
                yaw = math.atan2(target["y"] - y, target["x"] - x)
        return [yaw, pitch], fire

    def _glance(self, obs: mf.Observation) -> Optional[float]:
        best = None
        best_age = 1e9
        for s in obs.get("sounds") or []:
            if s.get("kind") not in ("gun", "mortar", "explosion", "laser", "lightning", "laser_charge"):
                continue
            age = float(s.get("age") or 0)
            if age < best_age:
                best_age = age
                best = float(s["bearing"])
        return best

    def _should_reload(self, me: Dict[str, Any], fight: bool) -> bool:
        if fight:
            return False
        w0 = _slot(me, 0)
        if not w0 or w0.get("kind") != "gun":
            return False
        if float(w0.get("reload") or 0) > 0:
            return False
        return float(w0.get("ammo") or 12) <= 3

    def _take(
        self,
        obs: mf.Observation,
        me: Dict[str, Any],
        x: float,
        y: float,
        z: float,
        threats: List[Dict[str, Any]],
    ) -> Optional[Tuple[float, Optional[float]]]:
        best = None
        best_r = -1
        for p in obs.get("pickups") or []:
            kind = p.get("kind")
            if kind not in LOOT_RANK or kind == "gun" or self._holding(me, kind):
                continue
            pos = p["pos"]
            if abs(float(pos[2]) - z) > 4:
                continue
            if math.hypot(float(pos[0]) - x, float(pos[1]) - y) > self.reach:
                continue
            rank = LOOT_RANK[kind]
            if rank > best_r:
                best_r = rank
                best = p
        if best is None:
            return None
        if _slot(me, 1) is None:
            return float(best["id"]), None
        held = _slot(me, 1)
        held_rank = LOOT_RANK.get(held.get("kind") if held else "", 0)
        if best_r > held_rank and not self._hot(threats, 55) and not (held and held.get("charge")):
            return float(best["id"]), 1
        return None

    def _pick_upgrade(self, offer: Dict[str, Any]) -> int:
        best_i = 0
        best_s = -1
        for i, choice in enumerate(offer.get("choices") or []):
            score = UPGRADE_RANK.get(choice.get("id"), 10)
            if score > best_s:
                best_s = score
                best_i = i
        return best_i

    def _label(self, me: Dict[str, Any], threats: List[Dict[str, Any]], fight: bool, urgent: bool) -> str:
        real = sum(1 for th in threats if not th.get("phantom"))
        state = "fight" if fight else "rot" if urgent else "hide"
        return f"{state} hp{me['hp']:.0f} n{real}"


if __name__ == "__main__":
    mf.run(Bot)
