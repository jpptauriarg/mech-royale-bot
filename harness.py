"""Local placement harness. Not the official server.

16 mechs. 15 hunters act as one squad: they share contact on our mech,
shoot only us, and otherwise stay in the shrinking zone. The player mech
is either oval-ghost (bot.py) or a kit-style baseline that walks to the
next circle and shoots the nearest hunter.

Score for a match is 16 - place, the same formula as the track.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
from scipy.sparse.csgraph import dijkstra

import bot as ghost
import world as worldmod

HERE = os.path.dirname(os.path.abspath(__file__))
HELLO = json.load(open(os.path.join(HERE, "tools", "hello.json")))
CFG = HELLO["config"]
MECH = CFG["mech"]
GUN = CFG["weapons"]["gun"]
ZONE = CFG["zone"]
WALK = float(MECH["walk_speed"])
BOOST = float(MECH["boost_speed"])
TURN = float(MECH["turn_rate"])
PITCH_RATE = float(MECH["pitch_rate"])
HP0 = float(MECH["hp"])
ENERGY0 = float(MECH["energy"])
REGEN = float(MECH["regen_per_s"])
REGEN_DELAY = float(MECH["regen_delay"])
BOOST_DRAIN = float(MECH["boost_drain"])
ENERGY_REGEN = float(MECH["energy_regen"])
GUN_DMG = float(GUN["damage"])
GUN_CD = float(GUN["cooldown"])
GUN_SPEED = float(GUN["speed"])
GUN_RANGE = float(GUN["range"])
GUN_MAG = int(GUN["mag"])
GUN_RELOAD = float(GUN["reload"])
FOV = math.radians(float(CFG["vision"]["fov_deg"]) / 2)
SIGHT = float(CFG["vision"]["range"])
NEAR = float(CFG["vision"]["near_range"])
DT = 0.1
MATCH_S = float(CFG["match"]["duration_s"])
SEP = float(CFG["match"]["spawn_min_separation"])
SPAWN_FRAC = float(CFG["match"]["spawn_zone_fraction"])


def angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Mech:
    def __init__(self, mid, x, y, z):
        self.id = mid
        self.x = x
        self.y = y
        self.z = z
        self.vx = 0.0
        self.vy = 0.0
        self.aim = 0.0
        self.pitch = 0.0
        self.cmd_aim = 0.0
        self.cmd_pitch = 0.0
        self.hp = HP0
        self.energy = ENERGY0
        self.alive = True
        self.ammo = GUN_MAG
        self.cd = 0.0
        self.reload = 0.0
        self.last_combat = -99.0
        self.path = []
        self.plan_t = -99.0
        self.boost = False
        self.mx = 0.0
        self.my = 0.0
        self.place = None
        self.death_t = None
        self.death_by = None
        self.kills = 0


class Bullet:
    def __init__(self, x, y, z, vx, vy, vz, owner, rng):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.owner = owner
        self.left = rng


def zone_schedule(rng):
    """List of phases: hold start, shrink start, shrink end, c0, r0, c1, r1, dps."""
    c = (float(ZONE["start_center"][0]), float(ZONE["start_center"][1]))
    r = float(ZONE["start_radius"])
    t = 0.0
    phases = []
    for ph in ZONE["phases"]:
        hold = float(ph["hold_s"])
        shrink = float(ph["shrink_s"])
        nr = float(ph["radius"])
        dps = float(ph["dps"])
        room = max(0.0, r - nr)
        ang = rng.random() * 2 * math.pi
        rad = room * math.sqrt(rng.random())
        nc = (c[0] + math.cos(ang) * rad, c[1] + math.sin(ang) * rad)
        phases.append({
            "hold_end": t + hold,
            "shrink_end": t + hold + shrink,
            "c0": c,
            "r0": r,
            "c1": nc,
            "r1": nr,
            "dps": dps,
        })
        t = t + hold + shrink
        c, r = nc, nr
    return phases


def zone_at(phases, t):
    for ph in phases:
        if t <= ph["shrink_end"] or ph is phases[-1]:
            if t <= ph["hold_end"]:
                u = 0.0
            elif t >= ph["shrink_end"]:
                u = 1.0
            else:
                span = ph["shrink_end"] - ph["hold_end"]
                u = (t - ph["hold_end"]) / span if span > 1e-6 else 1.0
            c = (
                ph["c0"][0] + u * (ph["c1"][0] - ph["c0"][0]),
                ph["c0"][1] + u * (ph["c1"][1] - ph["c0"][1]),
            )
            r = ph["r0"] + u * (ph["r1"] - ph["r0"])
            return c, r, ph
    ph = phases[-1]
    return ph["c1"], ph["r1"], ph


def obs_zone(phases, t):
    c, r, ph = zone_at(phases, t)
    return {
        "center": [c[0], c[1]],
        "radius": r,
        "next_center": [ph["c1"][0], ph["c1"][1]],
        "next_radius": ph["r1"],
        "shrink_start": ph["hold_end"],
        "shrink_end": ph["shrink_end"],
        "dps": ph["dps"],
    }


class Sim:
    def __init__(self, seed, policy):
        self.seed = seed
        self.policy = policy
        self.rng = random.Random(seed)
        self.phases = zone_schedule(random.Random(seed + 7919))
        self.W = worldmod.World(os.path.join(HERE, "maps.npz"))
        self.hello = json.loads(json.dumps(HELLO))
        self.hello["seed"] = seed
        self.hello["slot"] = 0
        self.bot = ghost.Bot(self.hello)
        self.mechs = self._spawn()
        self.player = self.mechs[0]
        self.bullets = []
        self.events = []
        self.pending_sounds = []
        self.t = 0.0
        self.tick = 0
        self.squad_x = self.player.x
        self.squad_y = self.player.y
        self.squad_t = -99.0
        self.anchors = [None] * 16
        self.anchor_phase = -1

    def _spawn(self):
        c0 = (float(ZONE["start_center"][0]), float(ZONE["start_center"][1]))
        limit = float(ZONE["start_radius"]) * SPAWN_FRAC
        free = np.flatnonzero(~self.bot.blk & self.bot.main)
        coords = []
        for n in free:
            x = float(self.bot.cx[n])
            y = float(self.bot.cy[n])
            if math.hypot(x - c0[0], y - c0[1]) <= limit:
                coords.append((x, y, int(n)))
        self.rng.shuffle(coords)
        chosen = []
        sep = SEP
        for min_sep in (sep, 100.0, 70.0, 40.0):
            chosen = []
            for x, y, _n in coords:
                if all(math.hypot(x - px, y - py) >= min_sep for px, py, _ in chosen):
                    chosen.append((x, y, _n))
                if len(chosen) == 16:
                    break
            if len(chosen) == 16:
                break
        if len(chosen) < 16:
            raise RuntimeError(f"only spawned {len(chosen)}")
        mechs = []
        for i, (x, y, _n) in enumerate(chosen):
            z = float(self.W.surf(x, y))
            mechs.append(Mech(i, x, y, z))
        return mechs

    def _alive(self):
        return [m for m in self.mechs if m.alive]

    def _integrate_aim(self, m):
        dy = angdiff(m.cmd_aim, m.aim)
        step = TURN * DT
        m.aim = wrap(m.aim + max(-step, min(step, dy)))
        dp = m.cmd_pitch - m.pitch
        step_p = PITCH_RATE * DT
        m.pitch = max(float(MECH["min_pitch"]), min(float(MECH["max_pitch"]), m.pitch + max(-step_p, min(step_p, dp))))

    def _move_mech(self, m, urgent_outside):
        spd_cap = BOOST if m.boost and m.energy > 1.0 else WALK
        length = math.hypot(m.mx, m.my)
        if length < 1e-4:
            m.vx *= 0.4
            m.vy *= 0.4
        else:
            scale = min(1.0, length)
            desired = spd_cap * scale
            m.vx = m.mx / length * desired
            m.vy = m.my / length * desired
            if m.boost and m.energy > 0.0 and scale > 0.15:
                m.energy = max(0.0, m.energy - BOOST_DRAIN * DT)
            elif m.energy < ENERGY0:
                m.energy = min(ENERGY0, m.energy + ENERGY_REGEN * DT)
        if length < 1e-4 and m.energy < ENERGY0:
            m.energy = min(ENERGY0, m.energy + ENERGY_REGEN * DT)
        nx = m.x + m.vx * DT
        ny = m.y + m.vy * DT
        if self.bot.free_at(nx, ny):
            m.x, m.y = nx, ny
        elif self.bot.free_at(nx, m.y):
            m.x = nx
            m.vy = 0.0
        elif self.bot.free_at(m.x, ny):
            m.y = ny
            m.vx = 0.0
        else:
            m.vx = m.vy = 0.0
        m.z = float(self.W.surf(m.x, m.y))
        if urgent_outside:
            pass

    def _blocked_step(self, x0, y0, z0, x1, y1, z1):
        return not bool(self.W.los([x0, y0, z0], [x1, y1, z1], canopy=False)[0])

    def _gun_state(self, m):
        return {
            "kind": "gun",
            "cooldown": max(0.0, m.cd),
            "ammo": m.ammo,
            "reload": max(0.0, m.reload),
            "stats": {
                "damage": GUN_DMG,
                "cooldown": GUN_CD,
                "speed": GUN_SPEED,
                "range": GUN_RANGE,
                "spread_deg": float(GUN["spread_deg"]),
                "mag": GUN_MAG,
                "reload": GUN_RELOAD,
            },
        }

    def _visible_to(self, eye, aim, others):
        if not others:
            return []
        feet = np.array([[m.x, m.y, m.z] for m in others], np.float64)
        seen = self.W.sees(np.tile(eye, (len(others), 1)), feet, canopy=True)
        out = []
        for m, ok in zip(others, seen):
            if not ok:
                continue
            d = math.hypot(m.x - eye[0], m.y - eye[1])
            if d <= NEAR or (d <= SIGHT and abs(angdiff(aim, math.atan2(m.y - eye[1], m.x - eye[0]))) <= FOV):
                out.append(m)
        return out

    def _player_obs(self, zone):
        p = self.player
        others = [m for m in self.mechs if m.alive and m.id != 0]
        vis = self._visible_to([p.x, p.y, p.z + 8.0], p.aim, others)
        mechs = []
        for m in vis:
            mechs.append({
                "id": m.id,
                "kind": "mech",
                "pos": [m.x, m.y, m.z],
                "vel": [m.vx, m.vy, 0.0],
                "aim": [m.aim, m.pitch],
                "hp": m.hp,
                "max_hp": HP0,
                "weapons": ["gun"],
                "charging": False,
            })
        c, r, _ph = zone_at(self.phases, self.t)
        proj = []
        for i, b in enumerate(self.bullets):
            if b.owner == 0:
                continue
            proj.append({
                "id": 1000 + i,
                "kind": "gun",
                "pos": [b.x, b.y, b.z],
                "vel": [b.vx, b.vy, b.vz],
            })
        return {
            "type": "obs",
            "tick": self.tick,
            "t": self.t,
            "alive": len(self._alive()),
            "self": {
                "id": 0,
                "pos": [p.x, p.y, p.z],
                "vel": [p.vx, p.vy, 0.0],
                "aim": [p.aim, p.pitch],
                "hp": p.hp,
                "max_hp": HP0,
                "energy": p.energy,
                "max_energy": ENERGY0,
                "grounded": True,
                "cubes": 0,
                "upgrades": [],
                "in_zone": math.hypot(p.x - c[0], p.y - c[1]) <= r,
                "weapons": [self._gun_state(p), None],
                "stats": {
                    "walk_speed": WALK,
                    "boost_speed": BOOST,
                    "turn_rate": TURN,
                    "eye_height": 8.0,
                    "muzzle_height": 7.0,
                },
            },
            "mechs": mechs,
            "projectiles": proj,
            "crates": [],
            "cubes": [],
            "pickups": [],
            "sounds": [],
            "beams": [],
            "events": list(self.events),
            "sounds": list(self.pending_sounds),
            "drops": [],
            "zone": zone,
            "offer": None,
        }

    def _apply_player(self, act):
        p = self.player
        move = act.get("move") or [0.0, 0.0]
        p.mx, p.my = float(move[0]), float(move[1])
        p.boost = bool(act.get("boost"))
        aim = act.get("aim") or [p.cmd_aim, p.cmd_pitch]
        p.cmd_aim = float(aim[0])
        p.cmd_pitch = float(aim[1])
        fire = act.get("fire") or [False, False]
        if act.get("reload") and p.reload <= 0.0 and p.ammo < GUN_MAG:
            p.reload = GUN_RELOAD
            p.ammo = 0
        if fire[0]:
            self._try_fire(p)

    def _try_fire(self, m):
        if m.cd > 0.0 or m.reload > 0.0 or m.ammo <= 0 or not m.alive:
            return False
        m.cd = GUN_CD
        m.ammo -= 1
        if m.ammo <= 0 and m.reload <= 0.0:
            m.reload = GUN_RELOAD
        cp = math.cos(m.pitch)
        dx = cp * math.cos(m.aim)
        dy = cp * math.sin(m.aim)
        dz = math.sin(m.pitch)
        mz = m.z + 7.0
        self.bullets.append(Bullet(
            m.x + dx * 3.0, m.y + dy * 3.0, mz + dz * 3.0,
            dx * GUN_SPEED, dy * GUN_SPEED, dz * GUN_SPEED,
            m.id, GUN_RANGE,
        ))
        m.last_combat = self.t
        m.fired = True
        return True

    def _plan_hunter(self, m, gx, gy):
        src = self.bot.node(m.x, m.y)
        dist, pred = dijkstra(self.bot.graph, indices=src, directed=False, return_predecessors=True, limit=1100.0)
        dst = self.bot.node(gx, gy)
        if not np.isfinite(dist[dst]):
            world = np.hypot(self.bot.cx - gx, self.bot.cy - gy)
            world[~np.isfinite(dist)] = 1e18
            dst = int(np.argmin(world))
            if not np.isfinite(dist[dst]):
                m.path = []
                return
        path = []
        n = int(dst)
        guard = 0
        while n != src and n >= 0 and guard < 4000:
            path.append(n)
            nxt = int(pred[n])
            if nxt < 0:
                path = []
                break
            n = nxt
            guard += 1
        path.reverse()
        m.path = path
        m.plan_t = self.t

    def _steer(self, m, gx, gy):
        while m.path and math.hypot(float(self.bot.cx[m.path[0]]) - m.x, float(self.bot.cy[m.path[0]]) - m.y) < 6.0:
            m.path.pop(0)
        if m.path:
            n = m.path[min(2, len(m.path) - 1)]
            return math.atan2(float(self.bot.cy[n]) - m.y, float(self.bot.cx[n]) - m.x)
        return math.atan2(gy - m.y, gx - m.x)

    def _hunter_think(self, zone):
        p = self.player
        c = (zone["center"][0], zone["center"][1])
        r = zone["radius"]
        phase_i = 0
        for i, ph in enumerate(self.phases):
            if self.t <= ph["shrink_end"] or ph is self.phases[-1]:
                phase_i = i
                break
        if phase_i != self.anchor_phase:
            self.anchor_phase = phase_i
            nx, ny = zone["next_center"]
            nr = max(8.0, float(zone["next_radius"]) * 0.55)
            for m in self.mechs[1:]:
                ang = (m.id / 15.0) * 2 * math.pi
                self.anchors[m.id] = (nx + math.cos(ang) * nr, ny + math.sin(ang) * nr)
        hunters = [m for m in self.mechs[1:] if m.alive]
        see = self._visible_to([p.x, p.y, p.z + 8.0], p.aim, hunters) if False else None
        # Hunters look with their own aim. Batch LOS from each hunter is heavier;
        # one query from the player is not the same. Do one sees() per hunter eye
        # batched as eyes x one target.
        if hunters and p.alive:
            eyes = np.array([[m.x, m.y, m.z + 8.0] for m in hunters], np.float64)
            feet = np.tile([p.x, p.y, p.z], (len(hunters), 1))
            los = self.W.sees(eyes, feet, canopy=True)
            for m, ok in zip(hunters, los):
                d = math.hypot(p.x - m.x, p.y - m.y)
                in_view = d <= NEAR or (d <= SIGHT and abs(angdiff(m.aim, math.atan2(p.y - m.y, p.x - m.x))) <= FOV)
                if ok and in_view:
                    self.squad_x, self.squad_y, self.squad_t = p.x, p.y, self.t
                    break
        know = p.alive and (self.t - self.squad_t) < 8.0
        player_in = math.hypot(p.x - c[0], p.y - c[1]) <= r + 12.0
        for m in hunters:
            outside = math.hypot(m.x - c[0], m.y - c[1]) > r - 6.0
            if outside:
                gx, gy = c
                chasing = False
            elif know and player_in:
                gx, gy = self.squad_x, self.squad_y
                chasing = True
            else:
                gx, gy = self.anchors[m.id]
                chasing = False
            if self.t - m.plan_t > 2.0 or not m.path:
                self._plan_hunter(m, gx, gy)
            yaw = self._steer(m, gx, gy)
            m.mx, m.my = math.cos(yaw), math.sin(yaw)
            dist_p = math.hypot(p.x - m.x, p.y - m.y)
            m.boost = chasing and m.energy > 28.0 and dist_p > 35.0 and not outside
            if outside:
                m.boost = m.energy > 8.0
            # Face the player while the squad has contact, else face travel.
            if know:
                m.cmd_aim = math.atan2(self.squad_y - m.y, self.squad_x - m.x)
                m.cmd_pitch = math.atan2((p.z + 4.5) - (m.z + 7.0), max(dist_p, 1.0))
            else:
                m.cmd_aim = yaw
                m.cmd_pitch = 0.0
            m._want_fire = False
            if know and dist_p < GUN_RANGE * 0.96:
                err = abs(angdiff(m.aim, m.cmd_aim))
                if err < math.atan2(3.0, max(dist_p, 1.0)) + 0.03:
                    eyes = [m.x, m.y, m.z + 7.0]
                    tgt = [p.x, p.y, p.z + 4.5]
                    if bool(self.W.los(eyes, tgt, canopy=False)[0]):
                        m._want_fire = True

    def _baseline_act(self, obs):
        me = obs["self"]
        x, y, z = me["pos"]
        zone = obs["zone"]
        goal = zone["next_center"]
        yaw = math.atan2(goal[1] - y, goal[0] - x)
        act = {
            "move": [math.cos(yaw), math.sin(yaw)],
            "boost": not me["in_zone"],
            "aim": [yaw, 0.0],
            "fire": [False, False],
        }
        enemies = obs["mechs"]
        if enemies:
            e = min(enemies, key=lambda m: math.hypot(m["pos"][0] - x, m["pos"][1] - y))
            tx, ty, tz = e["pos"]
            dist = math.hypot(tx - x, ty - y) or 1.0
            act["aim"] = [math.atan2(ty - y, tx - x), math.atan2((tz + 4.5) - (z + 7.0), dist)]
            act["fire"] = [True, False]
        return act

    def _step_weapons(self, m):
        if m.cd > 0.0:
            m.cd = max(0.0, m.cd - DT)
        if m.reload > 0.0:
            m.reload = max(0.0, m.reload - DT)
            if m.reload <= 0.0:
                m.ammo = GUN_MAG

    def _note_sounds(self):
        p = self.player
        if not p.alive:
            return
        for m in self.mechs[1:]:
            if not m.alive:
                continue
            d = math.hypot(m.x - p.x, m.y - p.y)
            kind = None
            reach = 0.0
            if getattr(m, "fired", False) and d <= 320.0:
                kind, reach = "gun", 320.0
            elif m.boost and math.hypot(m.vx, m.vy) > 4.0 and d <= 90.0:
                kind, reach = "boost", 90.0
            elif math.hypot(m.vx, m.vy) > 5.0 and d <= 40.0 and int(self.t * 10) % 6 == m.id % 6:
                kind, reach = "step", 40.0
            m.fired = False
            if kind is None:
                continue
            if d < 60.0:
                bucket = "near"
            elif d < 160.0:
                bucket = "mid"
            else:
                bucket = "far"
            bearing = math.atan2(m.y - p.y, m.x - p.x)
            bearing += math.radians(self.rng.gauss(0.0, 7.0))
            self.pending_sounds.append({
                "kind": kind,
                "bearing": bearing,
                "dist": bucket,
                "age": 0.05,
            })

    def _step_bullets(self):
        p = self.player
        kept = []
        hunters = [m for m in self.mechs[1:] if m.alive]
        for b in self.bullets:
            x1 = b.x + b.vx * DT
            y1 = b.y + b.vy * DT
            z1 = b.z + b.vz * DT
            travel = math.hypot(b.vx, b.vy, b.vz) * DT
            b.left -= travel
            if b.left <= 0.0 or self._blocked_step(b.x, b.y, b.z, x1, y1, z1):
                continue
            hit = None
            if b.owner == 0:
                best = 1e9
                for m in hunters:
                    # Distance from mech to the segment, in the horizontal plane, muzzle height.
                    vx, vy = x1 - b.x, y1 - b.y
                    seg2 = vx * vx + vy * vy
                    if seg2 < 1e-6:
                        continue
                    t = ((m.x - b.x) * vx + (m.y - b.y) * vy) / seg2
                    t = max(0.0, min(1.0, t))
                    cx = b.x + vx * t
                    cy = b.y + vy * t
                    if math.hypot(m.x - cx, m.y - cy) < 2.6 and abs((m.z + 4.5) - (b.z + (z1 - b.z) * t)) < 5.0:
                        if t < best:
                            best = t
                            hit = m
            else:
                vx, vy = x1 - b.x, y1 - b.y
                seg2 = vx * vx + vy * vy
                if seg2 > 1e-6:
                    t = ((p.x - b.x) * vx + (p.y - b.y) * vy) / seg2
                    t = max(0.0, min(1.0, t))
                    cx = b.x + vx * t
                    cy = b.y + vy * t
                    if p.alive and math.hypot(p.x - cx, p.y - cy) < 2.6 and abs((p.z + 4.5) - (b.z + (z1 - b.z) * t)) < 5.0:
                        hit = p
            if hit is not None and hit.alive:
                hit.hp -= GUN_DMG
                hit.last_combat = self.t
                owner = self.mechs[b.owner]
                owner.last_combat = self.t
                if hit is p:
                    bearing = math.atan2(b.y - p.y, b.x - p.x)
                    self.events.append({
                        "type": "damaged",
                        "amount": GUN_DMG,
                        "weapon": "gun",
                        "bearing": bearing,
                        "by": b.owner,
                    })
                if hit.hp <= 0.0 and hit.alive:
                    hit.death_by = "gun"
                    if b.owner == 0:
                        p.kills += 1
                continue
            b.x, b.y, b.z = x1, y1, z1
            kept.append(b)
        self.bullets = kept

    def _zone_and_regen(self, zone):
        c = (zone["center"][0], zone["center"][1])
        r = zone["radius"]
        dps = zone["dps"]
        for m in self.mechs:
            if not m.alive:
                continue
            inside = math.hypot(m.x - c[0], m.y - c[1]) <= r
            if not inside:
                m.hp -= dps * DT
                m.last_combat = self.t
                if m.hp <= 0.0:
                    m.death_by = "zone"
            elif self.t - m.last_combat >= REGEN_DELAY:
                m.hp = min(HP0, m.hp + REGEN * DT)

    def _assign_places(self):
        alive_before = [m for m in self.mechs if m.place is None]
        dying = [m for m in alive_before if m.hp <= 0.0 and m.alive]
        if not dying:
            return
        place = len(alive_before)
        for m in dying:
            m.alive = False
            m.place = place
            m.death_t = self.t

    def _finish_timeout(self):
        survivors = [m for m in self.mechs if m.alive]
        survivors.sort(key=lambda m: -m.hp)
        # Highest HP is the best place among survivors. Ties share the better place.
        i = 0
        next_place = 1
        while i < len(survivors):
            j = i
            while j < len(survivors) and survivors[j].hp == survivors[i].hp:
                j += 1
            for m in survivors[i:j]:
                m.place = next_place
                m.death_t = self.t
            next_place += j - i
            i = j

    def run(self):
        while self.t < MATCH_S and self.player.alive and sum(m.alive for m in self.mechs) > 1:
            zone = obs_zone(self.phases, self.t)
            # Events collected since the previous decision are part of this observation.
            obs = self._player_obs(zone)
            self.events = []
            self.pending_sounds = []
            if self.policy == "ghost":
                try:
                    act = self.bot.act(obs) or {}
                except Exception:
                    act = {"move": [0.0, 0.0], "boost": False, "aim": [self.player.aim, self.player.pitch], "fire": [False, False]}
            else:
                act = self._baseline_act(obs)
            self._apply_player(act)
            self._hunter_think(zone)
            for m in self.mechs:
                if not m.alive:
                    continue
                self._integrate_aim(m)
                self._step_weapons(m)
            # Hunters fire after their aim slews, using the aim they have this tick.
            for m in self.mechs[1:]:
                if m.alive and getattr(m, "_want_fire", False):
                    self._try_fire(m)
            for m in self.mechs:
                if m.alive:
                    c = (zone["center"][0], zone["center"][1])
                    outside = math.hypot(m.x - c[0], m.y - c[1]) > zone["radius"]
                    self._move_mech(m, outside)
            self._note_sounds()
            self._step_bullets()
            self._zone_and_regen(zone)
            self._assign_places()
            self.t += DT
            self.tick += 1
            if not self.player.alive:
                break
        if self.player.place is None:
            self._finish_timeout()
        hunters_before = 0
        pd = self.player.death_t if self.player.death_t is not None else 1e9
        for m in self.mechs[1:]:
            if m.death_t is not None and m.death_t < pd - 1e-6:
                hunters_before += 1
        zone_deaths = sum(1 for m in self.mechs if m.death_by == "zone")
        gun_deaths = sum(1 for m in self.mechs if m.death_by == "gun")
        return {
            "seed": self.seed,
            "policy": self.policy,
            "place": self.player.place,
            "points": 16 - self.player.place,
            "death_t": None if self.player.alive else self.player.death_t,
            "alive_end": self.player.alive,
            "kills": self.player.kills,
            "hunters_before": hunters_before,
            "death_by": self.player.death_by,
            "zone_deaths": zone_deaths,
            "gun_deaths": gun_deaths,
            "hp": round(self.player.hp, 1),
        }


def summarize(rows):
    pts = [r["points"] for r in rows]
    deaths = [r["death_t"] for r in rows if r["death_t"] is not None]
    return {
        "n": len(rows),
        "mean_points": sum(pts) / len(pts),
        "mean_place": sum(r["place"] for r in rows) / len(rows),
        "wins": sum(1 for r in rows if r["place"] == 1),
        "mean_death_t": (sum(deaths) / len(deaths)) if deaths else None,
        "survived": sum(1 for r in rows if r["alive_end"]),
        "mean_kills": sum(r["kills"] for r in rows) / len(rows),
        "mean_hunters_before": sum(r["hunters_before"] for r in rows) / len(rows),
        "death_by": {
            "gun": sum(1 for r in rows if r["death_by"] == "gun"),
            "zone": sum(1 for r in rows if r["death_by"] == "zone"),
            "alive": sum(1 for r in rows if r["death_by"] is None),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--policy", choices=("ghost", "baseline", "both"), default="both")
    args = ap.parse_args()
    policies = ["ghost", "baseline"] if args.policy == "both" else [args.policy]
    for policy in policies:
        rows = []
        t0 = time.perf_counter()
        for i in range(args.matches):
            seed = args.seed0 + i
            ts = time.perf_counter()
            row = Sim(seed, policy).run()
            rows.append(row)
            print(
                f"{policy} seed {seed} place {row['place']} points {row['points']} "
                f"death_t {row['death_t']} by {row['death_by']} kills {row['kills']} "
                f"hunters_before {row['hunters_before']} dt {time.perf_counter() - ts:.1f}s",
                flush=True,
            )
        summary = summarize(rows)
        summary["seconds"] = round(time.perf_counter() - t0, 1)
        print("SUMMARY", policy, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
