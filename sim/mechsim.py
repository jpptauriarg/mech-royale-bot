#!/usr/bin/env python3
"""A rough local stand-in for a Mech Royale practice match: our bot against 15 hunters.

It is NOT the real server. It exists to compare bot versions on the things that decide
placement: staying in the zone, not getting stuck, using buildings for cover, and fighting
hunters one at a time. Read the results as relative, not absolute.

What it models (numbers from the kit's recorded hello config):
- 20 Hz ticks, a decision every 2 ticks, the shrinking zone with its dps;
- walking/boosting with acceleration, step-up, walls (1 m height grid), jumping and jets;
- vision (near range, 120 deg wedge, line of sight, 5 s memory), sounds with bearing noise;
- gun bullets with travel time and spread, laser charge/release hitscan, lightning;
- rear armor, HP regen, floor weapon loot and the `take` action.
- the lightning "short" (a laser carrier it hits loses its charge for short_s).
What it leaves out: NPC drones, crates, cubes, upgrades, supply drops, mortars, trees,
bridges you walk under, lightning chains.

The hunters copy what the recorded match shows a house bot doing: grab a laser early, then
close to 50-130 m and trade gun and laser shots. They know roughly where we are (they "hunt us")
and never shoot each other.

Usage:
    python3 sim/mechsim.py --kit /path/to/mechfight-kit --matches 6 -- python3 -u bot.py
"""
import argparse
import gzip
import json
import math
import os
import random
import subprocess
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from arena import Arena, PathJob  # noqa: E402

DT = 0.05
TWO_PI = 2.0 * math.pi


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


def hyp(x, y):
    return math.hypot(x, y)


class Mech:
    def __init__(self, mid, x, y, z, cfg, hunter):
        m = cfg["mech"]
        self.id = mid
        self.x, self.y, self.z = x, y, z
        self.vx = self.vy = self.vz = 0.0
        self.grounded = True
        self.hp = float(m["hp"])
        self.max_hp = float(m["hp"])
        self.energy = float(m["energy"])
        self.max_energy = float(m["energy"])
        self.yaw = random.uniform(-math.pi, math.pi)
        self.pitch = 0.0
        self.cmd = {"move": [0, 0], "boost": False, "aim": [self.yaw, 0.0]}
        self.weapons = [new_weapon("gun", cfg), None]
        self.alive = True
        self.hunter = hunter
        self.last_combat = -99.0
        self.last_spend = -99.0
        self.step_t = 0.0
        self.events = []
        self.seen_t = {}  # other id -> last time we saw it
        self.place = None
        self.dmg_dealt = 0.0
        self.dmg_taken = 0.0
        self.kills = 0
        self.death_t = None
        self.charge_dir = None
        # hunter brain
        self.brain = None


def new_weapon(kind, cfg):
    st = dict(cfg["weapons"][kind])
    w = {"kind": kind, "cooldown": 0.0, "stats": st}
    if kind == "gun":
        w["ammo"] = st["mag"]
        w["reload"] = 0.0
    if kind == "laser":
        w["charge"] = 0.0
    return w


class Match:
    def __init__(self, cfg, sim_arena, nav_arena, seed, hunters_know=True, n_hunters=15, squad=False):
        self.cfg = cfg
        self.A = sim_arena
        self.N = nav_arena
        self.rng = random.Random(seed)
        random.seed(seed)
        self.t = 0.0
        self.tick = 0
        self.mechs = []
        self.bullets = []
        self.sounds = []
        self.pickups = {}
        self.hunters_know = hunters_know
        # squad: the house bots as one squad (shared vision; 1 hunter early, more as we kill
        # them, all of them late; the rest roam and fight only if they spot us).
        self.squad = squad
        self.squad_seen_t = -99.0
        self.next_id = 8000
        self._zone_setup()
        self._spawn(n_hunters + 1)
        self._loot()

    # ------------------------------------------------------------------ setup
    def _walkable_point(self, cx, cy, r, tries=400):
        A = self.A
        for _ in range(tries):
            ang = self.rng.uniform(0, TWO_PI)
            rr = r * math.sqrt(self.rng.random())
            x, y = A.clamp(cx + math.cos(ang) * rr, cy + math.sin(ang) * rr)
            k = A._idx(x, y)
            if A.bld[k] or A.cost[k] >= 1:
                continue
            return x, y
        return None

    def _spawn(self, n):
        zc = self.cfg["zone"]["start_center"]
        zr = self.cfg["zone"]["start_radius"] * self.cfg["match"]["spawn_zone_fraction"]
        sep = self.cfg["match"]["spawn_min_separation"]
        pts = []
        while len(pts) < n:
            p = self._walkable_point(zc[0], zc[1], zr)
            if p is None:
                continue
            if all(hyp(p[0] - q[0], p[1] - q[1]) >= sep for q in pts) or len(pts) > 0 and self.rng.random() < 0.002:
                pts.append(p)
        self.rng.shuffle(pts)
        for i, (x, y) in enumerate(pts):
            m = Mech(i, x, y, self.A.surface(x, y), self.cfg, hunter=(i != 0))
            self.mechs.append(m)
        self.me = self.mechs[0]
        for h in self.mechs[1:]:
            h.brain = Hunter(h, self)

    def _loot(self):
        lc = self.cfg["loot"]
        kinds = [k for k, w in lc["weights"].items() for _ in range(int(w * 10))]
        zc = self.cfg["zone"]["start_center"]
        for _ in range(int(lc["floor_count"])):
            p = self._walkable_point(zc[0], zc[1], self.cfg["zone"]["start_radius"])
            if p is None:
                continue
            kind = self.rng.choice(kinds)
            if kind == "mortar":
                kind = self.rng.choice(["laser", "lightning"])  # no mortar physics here
            self.pickups[self.next_id] = (kind, p[0], p[1], self.A.surface(*p))
            self.next_id += 1

    def _zone_setup(self):
        z = self.cfg["zone"]
        c = list(z["start_center"])
        r = float(z["start_radius"])
        t = 0.0
        self.phases = []
        for ph in z["phases"]:
            rn = float(ph["radius"])
            ang = self.rng.uniform(0, TWO_PI)
            off = (r - rn) * math.sqrt(self.rng.random()) * 0.9
            nc = [c[0] + math.cos(ang) * off, c[1] + math.sin(ang) * off]
            nc[0], nc[1] = self.A.clamp(nc[0], nc[1])
            if hyp(nc[0] - c[0], nc[1] - c[1]) + rn > r:
                nc = list(c)
            ss = t + ph["hold_s"]
            se = ss + ph["shrink_s"]
            self.phases.append((t, ss, se, tuple(c), r, tuple(nc), rn, ph["dps"]))
            c, r, t = nc, rn, se

    def zone(self):
        t = self.t
        for (t0, ss, se, c, r, nc, rn, dps) in self.phases:
            if t < se:
                if t <= ss:
                    cc, cr = c, r
                else:
                    f = (t - ss) / (se - ss)
                    cc = (c[0] + (nc[0] - c[0]) * f, c[1] + (nc[1] - c[1]) * f)
                    cr = r + (rn - r) * f
                return {"center": [round(cc[0], 2), round(cc[1], 2)], "radius": round(cr, 2),
                        "next_center": [round(nc[0], 2), round(nc[1], 2)], "next_radius": rn,
                        "shrink_start": ss, "shrink_end": se, "dps": dps}
        last = self.phases[-1]
        return {"center": list(last[5]), "radius": 0.0, "next_center": list(last[5]), "next_radius": 0.0,
                "shrink_start": last[1], "shrink_end": last[2], "dps": last[7]}

    # ------------------------------------------------------------------ perception
    def sees(self, a, b, extra=0.0):
        d = hyp(b.x - a.x, b.y - a.y)
        vis = self.cfg["vision"]
        if d > vis["range"] + extra:
            return False
        if d > vis["near_range"]:
            if abs(wrap(math.atan2(b.y - a.y, b.x - a.x) - a.yaw)) > math.radians(vis["fov_deg"]) / 2:
                return False
        eye = a.z + self.cfg["mech"]["eye_height"]
        return self.A.los(a.x, a.y, eye, b.x, b.y, b.z + 4.5) or self.A.los(a.x, a.y, eye, b.x, b.y, b.z + 8.8)

    def sound(self, kind, x, y, src):
        self.sounds.append((self.t, kind, x, y, src))

    def obs_for(self, m, last_t):
        cfg = self.cfg
        mem = cfg["vision"].get("memory_s", 5.0)
        mechs = []
        for o in self.mechs:
            if o is m or not o.alive:
                continue
            if self.sees(m, o):
                m.seen_t[o.id] = self.t
            if self.t - m.seen_t.get(o.id, -99) <= mem:
                w = [w["kind"] for w in o.weapons if w]
                mechs.append({"id": o.id, "kind": "mech", "pos": [round(o.x, 2), round(o.y, 2), round(o.z, 2)],
                              "vel": [round(o.vx, 2), round(o.vy, 2), round(o.vz, 2)],
                              "aim": [round(o.yaw, 3), round(o.pitch, 3)], "hp": round(o.hp, 2),
                              "max_hp": o.max_hp, "cubes": 0, "weapons": w,
                              "charging": any(w and w["kind"] == "laser" and w.get("charge", 0) > 0 for w in o.weapons)})
        projs = []
        for b in self.bullets:
            if b["owner"] != m.id and hyp(b["x"] - m.x, b["y"] - m.y) < 150.0:
                projs.append({"id": b["id"], "kind": "gun", "pos": [round(b["x"], 2), round(b["y"], 2), round(b["z"], 2)],
                              "vel": [round(b["vx"], 2), round(b["vy"], 2), round(b["vz"], 2)]})
        sounds = []
        rng_tab = cfg["sound"]["range"]
        noise = math.radians(cfg["sound"]["bearing_noise_deg"])
        for (st, kind, sx, sy, src) in self.sounds:
            if st <= last_t or src == m.id:
                continue
            d = hyp(sx - m.x, sy - m.y)
            if d > rng_tab.get(kind, 100):
                continue
            band = "near" if d < cfg["sound"]["near"] else ("mid" if d < cfg["sound"]["mid"] else "far")
            sounds.append({"kind": kind, "bearing": round(wrap(math.atan2(sy - m.y, sx - m.x) + self.rng.gauss(0, noise)), 3),
                           "dist": band, "age": round(self.t - st, 2)})
        pickups = []
        for pid, (kind, px, py, pz) in self.pickups.items():
            d = hyp(px - m.x, py - m.y)
            if d < 45 or (d < 200 and abs(wrap(math.atan2(py - m.y, px - m.x) - m.yaw)) < 1.05):
                if self.A.los(m.x, m.y, m.z + 8, px, py, pz + 0.5):
                    pickups.append({"id": pid, "kind": kind, "pos": [round(px, 2), round(py, 2), round(pz, 2)]})
        z = self.zone()
        in_zone = hyp(m.x - z["center"][0], m.y - z["center"][1]) <= z["radius"]
        st = dict(cfg["mech"])
        weps = []
        for w in m.weapons:
            if w is None:
                weps.append(None)
                continue
            d = {"kind": w["kind"], "cooldown": round(max(0.0, w["cooldown"]), 3), "stats": w["stats"]}
            if w["kind"] == "gun":
                d["ammo"] = w["ammo"]
                d["reload"] = round(max(0.0, w["reload"]), 3)
            if w["kind"] == "laser":
                d["charge"] = round(w.get("charge", 0.0), 3)
            weps.append(d)
        ev = m.events
        m.events = []
        return {"type": "obs", "tick": self.tick, "t": round(self.t, 2), "alive": sum(1 for o in self.mechs if o.alive),
                "self": {"id": m.id, "pos": [round(m.x, 2), round(m.y, 2), round(m.z, 2)],
                         "vel": [round(m.vx, 2), round(m.vy, 2), round(m.vz, 2)],
                         "aim": [round(m.yaw, 4), round(m.pitch, 4)], "hp": round(m.hp, 2), "max_hp": m.max_hp,
                         "energy": round(m.energy, 2), "max_energy": m.max_energy, "grounded": m.grounded,
                         "cubes": 0, "upgrades": [], "weapons": weps, "in_zone": in_zone, "stats": st},
                "mechs": mechs, "projectiles": projs, "crates": [], "cubes": [], "pickups": pickups,
                "sounds": sounds, "beams": [], "events": ev, "zone": z, "offer": None, "drops": []}

    def _assign(self):
        me = self.me
        alive = [h for h in self.mechs[1:] if h.alive]
        n = 15 if self.t > 330 else min(15, 1 + 2 * me.kills + int(self.t / 60.0))
        cur = [h for h in alive if h.brain.assigned]
        if len(cur) < n:
            free = sorted((h for h in alive if not h.brain.assigned), key=lambda h: hyp(h.x - me.x, h.y - me.y))
            for h in free[:n - len(cur)]:
                h.brain.assigned = True

    # ------------------------------------------------------------------ actions and physics
    def apply(self, m, act):
        for k in ("move", "boost", "aim"):
            if k in act:
                m.cmd[k] = act[k]
        m.fire = list(act.get("fire") or [False, False])[:2] + [False] * (2 - len(list(act.get("fire") or [])[:2]))
        m.jump = bool(act.get("jump"))
        if act.get("reload"):
            for w in m.weapons:
                if w and w["kind"] == "gun" and w["reload"] <= 0 and w["ammo"] < w["stats"]["mag"]:
                    w["reload"] = w["stats"]["reload"]
                    self.sound("reload", m.x, m.y, m.id)
        tk = act.get("take")
        if tk is not None and tk in self.pickups:
            kind, px, py, pz = self.pickups[tk]
            if hyp(px - m.x, py - m.y) <= self.cfg["loot"]["pickup_radius"] and abs(pz - m.z) <= 4:
                held = [w["kind"] for w in m.weapons if w]
                if kind not in held:
                    slot = act.get("drop")
                    if slot is None:
                        slot = 0 if m.weapons[0] is None else (1 if m.weapons[1] is None else None)
                    if slot is not None:
                        old = m.weapons[int(slot)]
                        nw = new_weapon(kind, self.cfg)
                        nw["cooldown"] = max(self.cfg["loot"]["equip_s"], nw["stats"].get("cooldown", 0))
                        m.weapons[int(slot)] = nw
                        del self.pickups[tk]
                        m.events.append({"type": "took", "id": tk, "kind": kind, "slot": int(slot)})
                        if not hasattr(m, "took_t"):
                            m.took_t = round(self.t, 1)
                        if old is not None:
                            self.pickups[self.next_id] = (old["kind"], m.x, m.y, m.z)
                            self.next_id += 1

    def physics(self, m):
        cfg = self.cfg["mech"]
        if not m.alive:
            return
        # aim slew
        ay, ap = float(m.cmd["aim"][0]), float(m.cmd["aim"][1])
        ap = max(cfg["min_pitch"], min(cfg["max_pitch"], ap))
        dyaw = wrap(ay - m.yaw)
        lim = cfg["turn_rate"] * DT
        m.yaw = wrap(m.yaw + max(-lim, min(lim, dyaw)))
        dp = ap - m.pitch
        lim = cfg["pitch_rate"] * DT
        m.pitch += max(-lim, min(lim, dp))
        mv = m.cmd["move"] or [0, 0]
        mx, my = float(mv[0]), float(mv[1])
        ml = hyp(mx, my)
        if ml > 1:
            mx, my = mx / ml, my / ml
            ml = 1.0
        charging = any(w and w["kind"] == "laser" and w.get("charge", 0) > 0 for w in m.weapons)
        boost = bool(m.cmd.get("boost")) and m.energy > 0
        spd = cfg["boost_speed"] if (boost and m.grounded and ml > 0.05) else cfg["walk_speed"]
        if charging:
            spd *= cfg["charge_move_mult"]
        dvx, dvy = mx * spd - m.vx, my * spd - m.vy
        acc = (cfg["ground_accel"] if m.grounded else cfg["air_accel"]) * DT
        dl = hyp(dvx, dvy)
        if dl > acc:
            dvx, dvy = dvx / dl * acc, dvy / dl * acc
        m.vx += dvx
        m.vy += dvy
        spent = False
        if boost and m.grounded and ml > 0.05:
            m.energy -= cfg["boost_drain"] * DT
            spent = True
        if m.jump and m.grounded:
            m.vz = cfg["jump_speed"]
            m.grounded = False
        if not m.grounded:
            if boost:
                m.vz = min(cfg["max_rise_speed"], m.vz + cfg["jet_accel"] * DT)
                m.energy -= cfg["jet_drain"] * DT
                spent = True
            m.vz -= cfg["gravity"] * DT
        if spent:
            m.last_spend = self.t
        elif self.t - m.last_spend > cfg["energy_regen_delay"]:
            m.energy = min(m.max_energy, m.energy + cfg["energy_regen"] * DT)
        m.energy = max(0.0, m.energy)
        # horizontal move with wall sliding
        nx, ny = m.x + m.vx * DT, m.y + m.vy * DT
        if self._blocked(nx, ny, m.z):
            if not self._blocked(nx, m.y, m.z):
                ny = m.y
                m.vy *= 0.2
            elif not self._blocked(m.x, ny, m.z):
                nx = m.x
                m.vx *= 0.2
            else:
                nx, ny = m.x, m.y
                m.vx *= 0.2
                m.vy *= 0.2
        m.x, m.y = self.A.clamp(nx, ny)
        # vertical
        floor = self._floor(m.x, m.y, m.z)
        m.z += m.vz * DT
        if m.z <= floor:
            m.z = floor
            m.vz = 0.0
            m.grounded = True
        elif m.grounded and m.z - floor < 1.3:
            m.z = floor  # walking down a slope
        else:
            m.grounded = False
        # steps
        hs = hyp(m.vx, m.vy)
        if m.grounded and hs > self.cfg["sound"]["step_speed"]:
            m.step_t += DT
            if m.step_t >= self.cfg["sound"]["step_every"]:
                m.step_t = 0.0
                self.sound("boost" if boost else "step", m.x, m.y, m.id)

    def _blocked(self, x, y, z):
        A = self.A
        lim = z + 1.2
        if A.surface(x, y) > lim:
            return True
        for k in range(10):
            a = k * (TWO_PI / 10)
            if A.surface(x + math.cos(a) * 2.3, y + math.sin(a) * 2.3) > lim:
                return True
        return False

    def _floor(self, x, y, z):
        A = self.A
        best = A.surface(x, y)
        for dx, dy in ((1.2, 0), (-1.2, 0), (0, 1.2), (0, -1.2)):
            s = A.surface(x + dx, y + dy)
            if s <= z + 1.2 and s > best:
                best = s
        return best

    def separate(self):
        ms = [m for m in self.mechs if m.alive]
        for i in range(len(ms)):
            a = ms[i]
            for j in range(i + 1, len(ms)):
                b = ms[j]
                dx, dy = b.x - a.x, b.y - a.y
                d = hyp(dx, dy)
                if 1e-6 < d < 5.0:
                    push = (5.0 - d) * 0.5
                    ux, uy = dx / d, dy / d
                    if not self._blocked(a.x - ux * push, a.y - uy * push, a.z):
                        a.x -= ux * push
                        a.y -= uy * push
                    if not self._blocked(b.x + ux * push, b.y + uy * push, b.z):
                        b.x += ux * push
                        b.y += uy * push

    # ------------------------------------------------------------------ weapons
    def weapons(self, m):
        if not m.alive:
            return
        for i, w in enumerate(m.weapons):
            if w is None:
                continue
            w["cooldown"] = max(0.0, w["cooldown"] - DT)
            trig = m.fire[i] if hasattr(m, "fire") else False
            k = w["kind"]
            st = w["stats"]
            if k == "gun":
                if w["reload"] > 0:
                    w["reload"] -= DT
                    if w["reload"] <= 0:
                        w["reload"] = 0.0
                        w["ammo"] = st["mag"]
                    continue
                if trig and w["cooldown"] <= 0 and w["ammo"] > 0:
                    w["cooldown"] = st["cooldown"]
                    w["ammo"] -= 1
                    self._shoot_gun(m, st)
                    if w["ammo"] <= 0:
                        w["reload"] = st["reload"]
            elif k == "laser":
                if w["charge"] > 0:
                    if trig:
                        w["charge"] += DT
                        if w["charge"] >= st["charge"] + st["hold_s"]:
                            self._fire_laser(m, w)
                    else:
                        if w["charge"] >= st["min_charge"]:
                            self._fire_laser(m, w)
                        else:
                            w["charge"] += DT  # fires as soon as min_charge is in
                elif trig and w["cooldown"] <= 0 and self.t >= getattr(m, "short_until", -1.0):
                    w["charge"] = DT
                    self.sound("laser_charge", m.x, m.y, m.id)
            elif k == "lightning":
                if trig and w["cooldown"] <= 0 and m.energy >= st["energy"]:
                    w["cooldown"] = st["cooldown"]
                    m.energy -= st["energy"]
                    m.last_spend = self.t
                    self._fire_lightning(m, st)

    def _muzzle(self, m):
        cp = math.cos(m.pitch)
        dx, dy, dz = cp * math.cos(m.yaw), cp * math.sin(m.yaw), math.sin(m.pitch)
        mz = m.z + self.cfg["mech"]["muzzle_height"]
        return m.x + dx * 3.0, m.y + dy * 3.0, mz + dz * 3.0, dx, dy, dz

    def _shoot_gun(self, m, st):
        sp = math.radians(st["spread_deg"])
        yaw = m.yaw + self.rng.gauss(0, sp)
        pitch = m.pitch + self.rng.gauss(0, sp)
        cp = math.cos(pitch)
        x, y, z, _, _, _ = self._muzzle(m)
        v = st["speed"]
        m.shots = getattr(m, "shots", {})
        m.shots["gun"] = m.shots.get("gun", 0) + 1
        self.bullets.append({"id": self.next_id, "owner": m.id, "x": x, "y": y, "z": z,
                             "vx": cp * math.cos(yaw) * v, "vy": cp * math.sin(yaw) * v, "vz": math.sin(pitch) * v,
                             "left": st["range"], "dmg": st["damage"]})
        self.next_id += 1
        self.sound("gun", m.x, m.y, m.id)
        m.last_combat_fire = self.t

    def _fire_laser(self, m, w):
        st = w["stats"]
        frac = st["min_frac"] + (1 - st["min_frac"]) * min(1.0, w["charge"] / st["charge"])
        m.shots = getattr(m, "shots", {})
        m.shots["laser"] = m.shots.get("laser", 0) + 1
        w["charge"] = 0.0
        w["cooldown"] = st["cooldown"]
        self.sound("laser", m.x, m.y, m.id)
        x, y, z, dx, dy, dz = self._muzzle(m)
        hit, dist = self._ray_hit(m, x, y, z, dx, dy, dz, st["range"])
        if hit is not None:
            self.damage(hit, m, st["damage"] * frac, "laser", x, y)

    def _fire_lightning(self, m, st):
        self.sound("lightning", m.x, m.y, m.id)
        m.shots = getattr(m, "shots", {})
        m.shots["lightning"] = m.shots.get("lightning", 0) + 1
        best = None
        bd = 1e9
        for o in self.mechs:
            if o is m or not o.alive or (m.hunter and o.hunter):
                continue
            d = hyp(o.x - m.x, o.y - m.y)
            if d > st["range"] or abs(wrap(math.atan2(o.y - m.y, o.x - m.x) - m.yaw)) > math.radians(st["cone_deg"]) / 2:
                continue
            if not self.A.los(m.x, m.y, m.z + 7, o.x, o.y, o.z + 4.5):
                continue
            if d < bd:
                bd, best = d, o
        if best is not None:
            self.damage(best, m, st["damage"], "lightning", m.x, m.y)
            # Short: a laser carrier loses its charge and can't start one for short_s.
            for w in best.weapons:
                if w and w["kind"] == "laser":
                    if w.get("charge", 0) > 0:
                        w["charge"] = 0.0
                    best.short_until = self.t + st.get("short_s", 2.0)
                    best.events.append({"type": "effect", "effect": "short", "by": m.id})

    def _ray_hit(self, m, x, y, z, dx, dy, dz, rng):
        # first wall along the ray
        A = self.A
        wall = rng
        n = int(rng)
        for k in range(1, n + 1):
            px, py, pz = x + dx * k, y + dy * k, z + dz * k
            if A.surface(px, py) > pz:
                wall = k
                break
        best = None
        bd = wall
        hl = hyp(dx, dy) or 1e-9
        for o in self.mechs:
            if o is m or not o.alive:
                continue
            # 2D ray-circle
            ox, oy = o.x - x, o.y - y
            tproj = (ox * dx + oy * dy) / (hl * hl)
            if tproj < 0:
                continue
            cx, cy = x + dx * tproj - o.x, y + dy * tproj - o.y
            off = hyp(cx, cy)
            if off > 2.5:
                continue
            tin = tproj - math.sqrt(max(0.0, 2.5 * 2.5 - off * off)) / hl
            zz = z + dz * tin
            if o.z - 0.2 <= zz <= o.z + 9.2 and tin < bd:
                bd, best = tin, o
        return best, bd

    def bullets_step(self):
        A = self.A
        keep = []
        for b in self.bullets:
            sx, sy, sz = b["x"], b["y"], b["z"]
            ex, ey, ez = sx + b["vx"] * DT, sy + b["vy"] * DT, sz + b["vz"] * DT
            seg = hyp(ex - sx, ey - sy)
            hit_t = 2.0
            n = max(1, int(seg))
            for k in range(1, n + 1):
                f = k / n
                if A.surface(sx + (ex - sx) * f, sy + (ey - sy) * f) > sz + (ez - sz) * f:
                    hit_t = f
                    break
            victim = None
            for o in self.mechs:
                if not o.alive or o.id == b["owner"]:
                    continue
                dx, dy = ex - sx, ey - sy
                l2 = dx * dx + dy * dy or 1e-9
                f = max(0.0, min(1.0, ((o.x - sx) * dx + (o.y - sy) * dy) / l2))
                if hyp(sx + dx * f - o.x, sy + dy * f - o.y) <= 2.8:
                    zz = sz + (ez - sz) * f
                    if o.z - 0.3 <= zz <= o.z + 9.3 and f < hit_t:
                        hit_t = f
                        victim = o
            if victim is not None:
                owner = self.mechs[b["owner"]]
                self.damage(victim, owner, b["dmg"], "gun", sx, sy)
                continue
            if hit_t <= 1.0:
                continue
            b["x"], b["y"], b["z"] = ex, ey, ez
            b["left"] -= seg
            if b["left"] > 0:
                keep.append(b)
        self.bullets = keep

    def damage(self, victim, src, amount, weapon, fx, fy):
        if not victim.alive:
            return
        if src is not None and src.hunter and victim.hunter:
            return  # house bots only hunt us
        mc = self.cfg["mech"]
        bearing = math.atan2(fy - victim.y, fx - victim.x)
        rear = abs(wrap(bearing - victim.yaw)) > math.radians(mc["front_arc_deg"]) / 2
        if rear and weapon != "zone":
            amount *= mc["rear_mult"]
        victim.hp -= amount
        victim.dmg_taken += amount
        victim.last_combat = self.t
        ev = {"type": "damaged", "amount": round(amount, 2), "weapon": weapon,
              "bearing": None if weapon == "zone" else round(bearing, 3),
              "by": None if src is None else src.id}
        if rear and weapon != "zone":
            ev["rear"] = True
        victim.events.append(ev)
        if src is not None:
            src.last_combat = self.t
            src.dmg_dealt += amount
            src.hits = getattr(src, "hits", {})
            src.hits[weapon] = src.hits.get(weapon, 0) + 1
            if victim.hunter and not src.hunter:
                victim.hit_by_us_t = self.t
            src.events.append({"type": "hit", "target": victim.id, "amount": round(amount, 2), "weapon": weapon})
        if victim.hp <= 0:
            victim.hp = 0.0
            victim.alive = False
            victim.death_t = self.t
            self.sound("death", victim.x, victim.y, victim.id)
            if src is not None:
                src.kills += 1
                src.events.append({"type": "kill", "target": victim.id})
            for w in victim.weapons:
                if w and w["kind"] != "gun":
                    self.pickups[self.next_id] = (w["kind"], victim.x, victim.y, victim.z)
                    self.next_id += 1

    def zone_and_regen(self):
        z = self.zone()
        mc = self.cfg["mech"]
        for m in self.mechs:
            if not m.alive:
                continue
            out = hyp(m.x - z["center"][0], m.y - z["center"][1]) > z["radius"]
            if out:
                self.damage(m, None, z["dps"] * DT, "zone", m.x, m.y)
                m.zone_taken = getattr(m, "zone_taken", 0.0) + z["dps"] * DT
                m.events = [e for e in m.events if not (e.get("weapon") == "zone" and e is not m.events[-1])]
            elif self.t - m.last_combat > mc["regen_delay"] and m.hp < m.max_hp:
                m.hp = min(m.max_hp, m.hp + mc["regen_per_s"] * DT)

    # ------------------------------------------------------------------ the loop
    def run(self, bot_cmd, bot_cwd, map_dir, hello, log=None, max_t=None):
        proc = subprocess.Popen(bot_cmd, cwd=bot_cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=1, text=True)
        h = dict(hello)
        h["slot"] = 0
        h["map"] = {"dir": map_dir, "files": ["v2/world.json", "v2/terrain.i16"]}
        proc.stdin.write(json.dumps(h, separators=(",", ":")) + "\n")
        proc.stdin.flush()
        ready = proc.stdout.readline()
        lat = []
        late = 0
        last_obs_t = -1.0
        max_t = max_t or self.cfg["match"]["duration_s"]
        dur = self.cfg["match"]["duration_s"]
        me = self.me
        trace = []
        while True:
            if self.tick % 2 == 0:
                # decisions
                if me.alive:
                    o = self.obs_for(me, last_obs_t)
                    last_obs_t = self.t
                    line = json.dumps(o, separators=(",", ":")) + "\n"
                    t0 = time.perf_counter()
                    proc.stdin.write(line)
                    proc.stdin.flush()
                    reply = proc.stdout.readline()
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    lat.append(dt_ms)
                    if not reply:
                        break
                    act = json.loads(reply)
                    if dt_ms > 50.0:
                        late += 1
                        act = {k: act[k] for k in ("move", "boost", "aim") if k in act}
                    self.apply(me, act)
                    if log is not None and self.tick % 20 == 0:
                        trace.append((round(self.t, 1), round(me.x, 1), round(me.y, 1), round(me.hp),
                                      sum(1 for m in self.mechs if m.alive),
                                      (act.get("debug") or {}).get("label", "")))
                if self.squad:
                    self._assign()
                for hm in self.mechs[1:]:
                    if hm.alive:
                        self.apply(hm, hm.brain.think())
            for m in self.mechs:
                self.physics(m)
            self.separate()
            for m in self.mechs:
                self.weapons(m)
            self.bullets_step()
            self.zone_and_regen()
            # placements
            alive = [m for m in self.mechs if m.alive]
            for m in self.mechs:
                if not m.alive and m.place is None:
                    m.place = len(alive) + 1
            self.sounds = [s for s in self.sounds if self.t - s[0] < 1.0]
            self.t += DT
            self.tick += 1
            if not me.alive or len(alive) <= 1 or self.t >= max_t:
                break
        if me.alive:
            if self.t >= dur or len(alive) <= 1:
                better = sum(1 for m in self.mechs if m.alive and m is not me and m.hp > me.hp)
                me.place = 1 + better
            else:
                me.place = None  # cut short
        try:
            proc.stdin.write(json.dumps({"type": "end", "place": me.place or 0, "kills": me.kills}) + "\n")
            proc.stdin.flush()
            proc.stdin.close()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
        lat.sort()
        res = {
            "place": me.place, "death_t": me.death_t, "kills": me.kills, "dealt": round(me.dmg_dealt),
            "taken": round(me.dmg_taken), "t_end": round(self.t, 1),
            "lat_p50": round(lat[len(lat) // 2], 2) if lat else 0, "lat_p99": round(lat[int(len(lat) * 0.99)], 2) if lat else 0,
            "lat_max": round(lat[-1], 2) if lat else 0, "late": late, "decisions": len(lat),
            "hunters_alive": sum(1 for m in self.mechs[1:] if m.alive),
            "zone_dmg": None,
            "loadout": [w["kind"] for w in me.weapons if w],
            "shots": getattr(me, "shots", {}), "hits": getattr(me, "hits", {}),
            "enemy_hp_left": sorted(round(m.hp) for m in self.mechs[1:] if m.alive and hasattr(m, "hit_by_us_t"))[:5],
            "took_t": getattr(me, "took_t", None),
            "zone_taken": round(getattr(me, "zone_taken", 0.0)),
        }
        if log is not None:
            res["trace"] = trace
        return res


class Hunter:
    """A house bot that hunts mech 0."""

    def __init__(self, m, match):
        self.m = m
        self.M = match
        r = match.rng
        self.pref = r.uniform(50.0, 130.0)
        self.intel_t = -99.0
        self.known = None
        self.path = None
        self.path_goal = None
        self.path_t = -99.0
        self.path_i = 0
        self.strafe = 1.0 if r.random() < 0.5 else -1.0
        self.strafe_t = 0.0
        self.loot_goal = None
        self.stuck = 0
        self.last = (m.x, m.y)
        self.skill = r.uniform(0.6, 1.0)
        self.assigned = not match.squad
        self.roam = None

    def think(self):
        m, M = self.m, self.M
        tgt = M.me
        cfg = M.cfg
        t = M.t
        act = {"move": [0, 0], "boost": False, "fire": [False, False]}
        see = tgt.alive and M.sees(m, tgt)
        if see:
            self.known = (tgt.x, tgt.y, tgt.z, tgt.vx, tgt.vy)
            self.intel_t = t
            if M.squad:
                M.squad_seen_t = t
        elif M.squad and self.assigned and t - M.squad_seen_t < 0.15:
            self.known = (tgt.x, tgt.y, tgt.z, tgt.vx, tgt.vy)  # a squadmate sees it
            self.intel_t = t
        elif M.squad and not self.assigned:
            pass
        elif M.hunters_know and t - self.intel_t > 2.0:
            n = 12.0
            self.known = (tgt.x + M.rng.gauss(0, n), tgt.y + M.rng.gauss(0, n), tgt.z, 0.0, 0.0)
            self.intel_t = t
        z = M.zone()
        goal = None
        # zone first
        nc, nr = z["next_center"], z["next_radius"]
        dn = hyp(m.x - nc[0], m.y - nc[1])
        if (dn > nr * 0.85 and t > z["shrink_start"] - 25) or hyp(m.x - z["center"][0], m.y - z["center"][1]) > z["radius"] - 5:
            goal = (nc[0], nc[1])
        has_laser = any(w and w["kind"] == "laser" for w in m.weapons)
        if goal is None and not has_laser and not see and t < 60:
            best, bd = None, 90.0
            for pid, (kind, px, py, pz) in M.pickups.items():
                if kind != "laser":
                    continue
                d = hyp(px - m.x, py - m.y)
                if d < bd:
                    bd, best = d, (pid, px, py)
            if best is not None:
                goal = (best[1], best[2])
                if bd < 3.5:
                    act["take"] = best[0]
        d_t = hyp(tgt.x - m.x, tgt.y - m.y)
        if see:
            # aim with lead, fire
            st = m.weapons[0]["stats"] if m.weapons[0] else cfg["weapons"]["gun"]
            fl = d_t / st["speed"]
            lag = 0.3 * (1.2 - self.skill)
            px = tgt.x + tgt.vx * (fl - lag * 0.0) * self.skill
            py = tgt.y + tgt.vy * (fl) * self.skill
            charging = any(w and w["kind"] == "laser" and w.get("charge", 0) > 0 for w in m.weapons)
            if charging:
                px, py = tgt.x + tgt.vx * 0.05, tgt.y + tgt.vy * 0.05
            yaw = math.atan2(py - m.y, px - m.x)
            pitch = math.atan2((tgt.z + 4.5) - (m.z + 7.0), max(1.0, hyp(px - m.x, py - m.y)))
            act["aim"] = [yaw, pitch]
            err = abs(wrap(m.yaw - yaw))
            tol = math.atan2(2.2, max(d_t, 1.0))
            for i, w in enumerate(m.weapons):
                if not w:
                    continue
                if w["kind"] == "gun" and err < tol * 1.5 and d_t < w["stats"]["range"]:
                    act["fire"][i] = True
                elif w["kind"] == "laser":
                    if w["charge"] > 0:
                        act["fire"][i] = not (w["charge"] >= w["stats"]["charge"] and err < tol)
                    elif w["cooldown"] <= 0 and err < 0.3 and d_t < w["stats"]["range"]:
                        act["fire"][i] = True
                elif w["kind"] == "lightning" and d_t < 44 and err < 0.25:
                    act["fire"][i] = True
            if goal is None:
                if d_t > self.pref + 25:
                    goal = (tgt.x, tgt.y)
                elif d_t < self.pref - 30:
                    ux, uy = (m.x - tgt.x) / max(d_t, 1), (m.y - tgt.y) / max(d_t, 1)
                    act["move"] = [ux, uy]
                else:
                    if t > self.strafe_t:
                        self.strafe = -self.strafe
                        self.strafe_t = t + M.rng.uniform(1.0, 3.0)
                    ux, uy = (tgt.x - m.x) / max(d_t, 1), (tgt.y - m.y) / max(d_t, 1)
                    act["move"] = [-uy * self.strafe, ux * self.strafe]
        elif M.squad and not self.assigned:
            # roam inside the zone
            if self.roam is None or hyp(self.roam[0] - m.x, self.roam[1] - m.y) < 10:
                zc = z["next_center"]
                rr = max(10.0, z["next_radius"] * 0.8)
                a_ = M.rng.uniform(0, TWO_PI)
                self.roam = (zc[0] + math.cos(a_) * rr * M.rng.random(), zc[1] + math.sin(a_) * rr * M.rng.random())
            if goal is None:
                goal = self.roam
            act["aim"] = [math.atan2(goal[1] - m.y, goal[0] - m.x), 0.0]
        else:
            if self.known is not None:
                kx, ky = self.known[0], self.known[1]
                act["aim"] = [math.atan2(ky - m.y, kx - m.x), 0.0]
                if goal is None:
                    goal = (kx, ky)
            # release a held laser charge somewhere harmless only when forced
            for i, w in enumerate(m.weapons):
                if w and w["kind"] == "laser" and w.get("charge", 0) > 0:
                    act["fire"][i] = w["charge"] < w["stats"]["charge"] + w["stats"]["hold_s"] - 0.2
        if goal is not None:
            mv = self._nav(goal)
            act["move"] = mv
        # reload when empty and not seeing
        for w in m.weapons:
            if w and w["kind"] == "gun" and w["ammo"] < 4 and not see and w["reload"] <= 0:
                act["reload"] = True
        # stuck handling
        moved = hyp(m.x - self.last[0], m.y - self.last[1])
        self.last = (m.x, m.y)
        if hyp(*act["move"]) > 0.3 and moved < 0.2:
            self.stuck += 1
        else:
            self.stuck = max(0, self.stuck - 1)
        if self.stuck > 10:
            act["jump"] = True
            act["boost"] = True
            self.path = None
            if self.stuck > 30:
                self.stuck = 0
        if not m.grounded and m.energy > 20 and self.stuck > 5:
            act["boost"] = True
        return act

    def _nav(self, goal):
        m, M = self.m, self.M
        N = M.N
        gx, gy = N.clamp(*goal)
        d = hyp(gx - m.x, gy - m.y)
        if d < 2.0:
            return [0.0, 0.0]
        if d < 50 and N.walkable(m.x, m.y, gx, gy, strict=False):
            return [(gx - m.x) / d, (gy - m.y) / d]
        if (self.path is None or M.t - self.path_t > 3.0
                or hyp(self.path_goal[0] - gx, self.path_goal[1] - gy) > 25.0):
            job = PathJob(N, m.x, m.y, gx, gy, max_nodes=6000, weight=1.8)
            while not job.step(6000):
                pass
            self.path = job.path
            self.path_goal = (gx, gy)
            self.path_t = M.t
            self.path_i = 0
        path = self.path
        if not path:
            return [(gx - m.x) / d, (gy - m.y) / d]
        bi, bd = self.path_i, 1e9
        for k in range(self.path_i, min(len(path), self.path_i + 20)):
            dd = hyp(path[k][0] - m.x, path[k][1] - m.y)
            if dd < bd:
                bd, bi = dd, k
        self.path_i = bi
        tgt = path[min(len(path) - 1, bi + 2)]
        for k in range(min(len(path) - 1, bi + 10), bi, -1):
            if N.walkable(m.x, m.y, path[k][0], path[k][1], strict=False):
                tgt = path[k]
                break
        dx, dy = tgt[0] - m.x, tgt[1] - m.y
        dl = hyp(dx, dy) or 1.0
        return [dx / dl, dy / dl]


# ---------------------------------------------------------------------- runner
_W = {}


def _init(kit, cell):
    _W["sim"] = Arena(os.path.join(kit, "arena"), cell=1.0)
    _W["nav"] = Arena(os.path.join(kit, "arena"), cell=2.0)
    with gzip.open(os.path.join(kit, "check", "stream.jsonl.gz"), "rt") as f:
        _W["hello"] = json.loads(f.readline())


def _one(args):
    seed, cmd, cwd, kit, know, trace, nh, squad = args
    hello = _W["hello"]
    m = Match(hello["config"], _W["sim"], _W["nav"], seed, hunters_know=know, n_hunters=nh, squad=squad)
    t0 = time.perf_counter()
    res = m.run(cmd, cwd, os.path.join(kit, "arena"), hello, log=[] if trace else None)
    res["seed"] = seed
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser(usage="%(prog)s [options] -- <bot command...>")
    ap.add_argument("--kit", required=True, help="path to the mechfight kit (arena/ and check/stream.jsonl.gz)")
    ap.add_argument("--matches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--hunters", type=int, default=15)
    ap.add_argument("--blind", action="store_true", help="hunters only know what they see")
    ap.add_argument("--squad", action="store_true", help="house bots as one squad: shared vision, 1 hunter early, more per kill, all late")
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--json", default=None, help="write results here")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    jobs = [(a.seed + i, cmd, os.path.abspath(a.cwd), os.path.abspath(a.kit), not a.blind, a.trace, a.hunters, a.squad)
            for i in range(a.matches)]
    with Pool(a.jobs, initializer=_init, initargs=(os.path.abspath(a.kit), 1.0)) as pool:
        results = pool.map(_one, jobs)
    pts = []
    for r in results:
        place = r["place"] if r["place"] is not None else 16
        pts.append(16 - place)
        print("seed %d  place %s  died %s  kills %d  dealt %d  taken %d  hunters_left %d  lat p99 %.1f max %.1f ms late %d  (%.0fs)"
              % (r["seed"], r["place"], r["death_t"] and round(r["death_t"], 1), r["kills"], r["dealt"], r["taken"],
                 r["hunters_alive"], r["lat_p99"], r["lat_max"], r["late"], r["wall_s"]))
        if a.trace:
            for row in r["trace"]:
                print("   ", row)
    print("MEAN SCORE %.2f over %d matches (mean place %.2f)" % (sum(pts) / len(pts), len(pts), 16 - sum(pts) / len(pts)))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
