"""Local placement harness.

This is not the official simulator. It uses the kit map and the hello config from
check/stream.jsonl.gz, then runs a simplified ground fight: accelerated movement,
building collision, gun bullets with travel time, vision, zone damage, and regen.
It measures mean (16 - place) for shade against baseline-like opponents.

Modes
  ffa             shade vs 15 copies of the kit example policy
  squad           shade vs 15 cooperating house-style bots (they hunt shade, not each other)
  baseline-ffa    one kit example vs 15 other kit examples (symmetry check, ~7.5)
  baseline-squad  one kit example vs the same 15 cooperating bots
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import nav  # noqa: E402
from bot import Bot, ang_diff  # noqa: E402

with gzip.open(os.path.join(ROOT, "check", "stream.jsonl.gz"), "rt") as _f:
    CFG = json.loads(_f.readline())["config"]

MECH = CFG["mech"]
GUN = CFG["weapons"]["gun"]
WALK = float(MECH["walk_speed"])
BOOST = float(MECH["boost_speed"])
ACCEL = float(MECH["ground_accel"])
GUN_SPEED = float(GUN["speed"])
GUN_RANGE = float(GUN["range"])
GUN_DMG = float(GUN["damage"])
GUN_CD = float(GUN["cooldown"])
GUN_MAG = int(GUN["mag"])
GUN_RELOAD = float(GUN["reload"])
REGEN_DELAY = float(MECH["regen_delay"])
REGEN = float(MECH["regen_per_s"])
BOOST_DRAIN = float(MECH["boost_drain"])
ENERGY_REGEN = float(MECH["energy_regen"])
ENERGY_DELAY = float(MECH["energy_regen_delay"])
REAR = float(MECH["rear_mult"])
HALF_ARC = math.radians(float(MECH["front_arc_deg"]) / 2)
DURATION = float(CFG["match"]["duration_s"])
DT = 0.1
FOV = math.radians(float(CFG["vision"]["fov_deg"]) / 2)
SEE = float(CFG["vision"]["range"])
NEAR = float(CFG["vision"]["near_range"])
MEMORY = float(CFG["vision"].get("memory_s") or 5)
SEP = float(CFG["match"]["spawn_min_separation"])
ARENA_DIR = os.path.join(ROOT, "arena")


def fast_los(arena: nav.Arena, x0: float, y0: float, x1: float, y1: float) -> bool:
    """Grid DDA. Every crossed cell counts, so thin buildings block sight."""
    dx = x1 - x0
    dy = y1 - y0
    if dx * dx + dy * dy < 144.0:
        return True
    cell = arena.cell
    minx = arena.minx
    miny = arena.miny
    nx = arena.nx
    ny = arena.ny
    blocked = arena.blocked
    i = int((x0 - minx) / cell)
    j = int((y0 - miny) / cell)
    i_end = int((x1 - minx) / cell)
    j_end = int((y1 - miny) / cell)
    step_i = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_j = 1 if dy > 0 else (-1 if dy < 0 else 0)
    inv_dx = 1.0 / dx if dx != 0.0 else 0.0
    inv_dy = 1.0 / dy if dy != 0.0 else 0.0
    if step_i > 0:
        t_max_x = ((i + 1) * cell + minx - x0) * inv_dx
    elif step_i < 0:
        t_max_x = (i * cell + minx - x0) * inv_dx
    else:
        t_max_x = 1e30
    if step_j > 0:
        t_max_y = ((j + 1) * cell + miny - y0) * inv_dy
    elif step_j < 0:
        t_max_y = (j * cell + miny - y0) * inv_dy
    else:
        t_max_y = 1e30
    t_dx = abs(cell * inv_dx) if step_i else 1e30
    t_dy = abs(cell * inv_dy) if step_j else 1e30
    for _ in range(96):
        if i == i_end and j == j_end:
            return True
        if t_max_x < t_max_y:
            t_max_x += t_dx
            i += step_i
        else:
            t_max_y += t_dy
            j += step_j
        if i == i_end and j == j_end:
            return True
        if i < 0 or j < 0 or i >= nx or j >= ny or blocked[j * nx + i]:
            return False
    return True


class Fighter:
    def __init__(self, fid: int, x: float, y: float, z: float, kind: str):
        self.id = fid
        self.x = x
        self.y = y
        self.z = z
        self.vx = 0.0
        self.vy = 0.0
        self.hp = float(MECH["hp"])
        self.max_hp = float(MECH["hp"])
        self.energy = float(MECH["energy"])
        self.max_energy = float(MECH["energy"])
        self.aim = 0.0
        self.pitch = 0.0
        self.alive = True
        self.kind = kind
        self.team = 1 if kind == "squad" else 0
        self.ammo = GUN_MAG
        self.reload = 0.0
        self.gun_cd = 0.0
        self.cubes = 0
        self.armor = 0.0
        self.last_damage = -1e9
        self.last_spend = -1e9
        self.next_step = 0.0
        self.seen = {}
        self.place = None
        self.cause = None
        self.death_t = None
        self.boosting = False
        self.bot = None


class Bullet:
    def __init__(self, owner: int, team: int, x: float, y: float, z: float, vx: float, vy: float, vz: float):
        self.owner = owner
        self.team = team
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.left = GUN_RANGE


class CubePile:
    def __init__(self, x: float, y: float, z: float, n: int):
        self.x = x
        self.y = y
        self.z = z
        self.n = n


class Zone:
    def __init__(self, rng: random.Random):
        zcfg = CFG["zone"]
        c = [float(zcfg["start_center"][0]), float(zcfg["start_center"][1])]
        r = float(zcfg["start_radius"])
        t = 0.0
        self.phases = []
        for ph in zcfg["phases"]:
            nr = float(ph["radius"])
            max_off = max(0.0, r - nr)
            ang = rng.random() * math.tau
            rad = max_off * math.sqrt(rng.random()) * 0.85
            nc = [c[0] + math.cos(ang) * rad, c[1] + math.sin(ang) * rad]
            shrink_start = t + float(ph["hold_s"])
            shrink_end = shrink_start + float(ph["shrink_s"])
            self.phases.append(
                {
                    "c0": c[:],
                    "r0": r,
                    "c1": nc[:],
                    "r1": nr,
                    "shrink_start": shrink_start,
                    "shrink_end": shrink_end,
                    "dps": float(ph["dps"]),
                }
            )
            t = shrink_end
            c, r = nc, nr
        self.end_t = t

    def at(self, t: float):
        ph = self.phases[-1]
        for item in self.phases:
            ph = item
            if t <= item["shrink_end"]:
                break
        if t <= ph["shrink_start"]:
            c, r = ph["c0"], ph["r0"]
        else:
            span = max(0.01, ph["shrink_end"] - ph["shrink_start"])
            u = min(1.0, max(0.0, (t - ph["shrink_start"]) / span))
            c = [ph["c0"][0] + (ph["c1"][0] - ph["c0"][0]) * u, ph["c0"][1] + (ph["c1"][1] - ph["c0"][1]) * u]
            r = ph["r0"] + (ph["r1"] - ph["r0"]) * u
        return {
            "center": [c[0], c[1]],
            "radius": r,
            "next_center": [ph["c1"][0], ph["c1"][1]],
            "next_radius": ph["r1"],
            "shrink_start": ph["shrink_start"],
            "shrink_end": ph["shrink_end"],
            "dps": ph["dps"],
        }


def spawn_points(rng: random.Random, arena: nav.Arena, n: int = 16):
    zc = CFG["zone"]["start_center"]
    zr = float(CFG["zone"]["start_radius"]) * float(CFG["match"]["spawn_zone_fraction"])
    pts = []
    sep = SEP
    for _relax in range(4):
        pts = []
        for _i in range(n):
            placed = False
            for _try in range(500):
                ang = rng.random() * math.tau
                rad = zr * math.sqrt(rng.random())
                x = float(zc[0]) + math.cos(ang) * rad
                y = float(zc[1]) + math.sin(ang) * rad
                if arena.blocked_at(x, y):
                    continue
                if any((x - px) ** 2 + (y - py) ** 2 < sep * sep for px, py in pts):
                    continue
                pts.append((x, y))
                placed = True
                break
            if not placed:
                break
        if len(pts) == n:
            return pts
        sep *= 0.75
    while len(pts) < n:
        ang = rng.random() * math.tau
        rad = zr * math.sqrt(rng.random())
        x = float(zc[0]) + math.cos(ang) * rad
        y = float(zc[1]) + math.sin(ang) * rad
        if not arena.blocked_at(x, y):
            pts.append((x, y))
    return pts[:n]


def can_see(arena, a: Fighter, b: Fighter) -> bool:
    dx = b.x - a.x
    dy = b.y - a.y
    d2 = dx * dx + dy * dy
    if d2 > SEE * SEE:
        return False
    if not fast_los(arena, a.x, a.y, b.x, b.y):
        return False
    if d2 <= NEAR * NEAR:
        return True
    if abs(ang_diff(math.atan2(dy, dx), a.aim)) <= FOV:
        return True
    return False


def visible_ids(arena, agents, viewer: Fighter, now: float):
    out = []
    for other in agents:
        if other is viewer or not other.alive:
            continue
        if can_see(arena, viewer, other):
            viewer.seen[other.id] = now
        seen_t = viewer.seen.get(other.id)
        if seen_t is not None and now - seen_t <= MEMORY:
            out.append(other)
    return out


def baseline_action(me: Fighter, enemies, zone, tick: int):
    yaw = math.atan2(zone["next_center"][1] - me.y, zone["next_center"][0] - me.x)
    act = {
        "move": [math.cos(yaw), math.sin(yaw)],
        "aim": [yaw, 0.0],
        "boost": math.hypot(me.x - zone["center"][0], me.y - zone["center"][1]) > zone["radius"],
        "jump": tick % 40 == 0,
        "fire": [False, False],
    }
    if enemies:
        target = min(enemies, key=lambda o: (o.x - me.x) ** 2 + (o.y - me.y) ** 2)
        dist = math.hypot(target.x - me.x, target.y - me.y) or 1.0
        aim = math.atan2(target.y - me.y, target.x - me.x)
        pitch = math.atan2((target.z + 4.5) - (me.z + 7.0), dist)
        act["aim"] = [aim, pitch]
        act["fire"] = [True, False]
    return act


def squad_action(me: Fighter, enemies, zone, shared, arena, tick: int):
    prey = None
    for o in enemies:
        if o.id == 0:
            prey = o
            shared["prey"] = (o.x, o.y, shared["t"])
            break
    cx, cy = zone["center"]
    radius = max(0.0, zone["radius"])
    ang = shared["orbit"] + me.id * 0.55
    stand = max(0.0, radius - 28.0) * 0.62
    ax = cx + math.cos(ang) * stand
    ay = cy + math.sin(ang) * stand
    if "prey" in shared and shared["t"] - shared["prey"][2] < 6.0:
        px, py, _pt = shared["prey"]
        # Hunt, but keep the anchor inside the safe circle.
        hx, hy = px - cx, py - cy
        hd = math.hypot(hx, hy) or 1.0
        cap = max(0.0, radius - 20.0)
        if hd > cap:
            px = cx + hx / hd * cap
            py = cy + hy / hd * cap
        ax = ax * 0.35 + px * 0.65
        ay = ay * 0.35 + py * 0.65
    mx, my = ax - me.x, ay - me.y
    # Spread so fifteen mechs are not one bullet magnet.
    mx += math.cos(ang + 1.2) * 0.35
    my += math.sin(ang + 1.2) * 0.35
    mx, my = arena.avoid(me.x, me.y, mx, my)
    mag = math.hypot(mx, my) or 1.0
    outside = math.hypot(me.x - cx, me.y - cy) > radius
    act = {
        "move": [mx / mag, my / mag],
        "aim": [math.atan2(my, mx), 0.0],
        "boost": outside or (prey is not None and math.hypot(prey.x - me.x, prey.y - me.y) < 80),
        "jump": False,
        "fire": [False, False],
    }
    if prey is not None:
        dist = math.hypot(prey.x - me.x, prey.y - me.y) or 1.0
        tof = dist / GUN_SPEED
        aim = math.atan2(prey.y + prey.vy * tof - me.y, prey.x + prey.vx * tof - me.x)
        pitch = math.atan2((prey.z + 4.5) - (me.z + 7.0), dist)
        act["aim"] = [aim, max(-0.7, min(1.2, pitch))]
        if dist < GUN_RANGE:
            act["fire"] = [True, False]
    return act


def make_obs(me: Fighter, enemies, zone, now: float, tick: int, projectiles, sounds, events, cubes):
    mechs = []
    for o in enemies:
        mechs.append(
            {
                "id": o.id,
                "kind": "mech",
                "pos": [o.x, o.y, o.z],
                "vel": [o.vx, o.vy, 0.0],
                "aim": [o.aim, o.pitch],
                "hp": o.hp,
                "max_hp": o.max_hp,
                "cubes": o.cubes,
                "weapons": ["gun"],
                "charging": False,
            }
        )
    prots = []
    for b in projectiles:
        if b.owner == me.id:
            continue
        if (b.x - me.x) ** 2 + (b.y - me.y) ** 2 > SEE * SEE:
            continue
        prots.append({"id": id(b) % 100000, "kind": "gun", "pos": [b.x, b.y, b.z], "vel": [b.vx, b.vy, b.vz]})
    return {
        "type": "obs",
        "tick": tick,
        "t": now,
        "alive": None,  # filled by caller
        "self": {
            "id": me.id,
            "pos": [me.x, me.y, me.z],
            "vel": [me.vx, me.vy, 0.0],
            "aim": [me.aim, me.pitch],
            "hp": me.hp,
            "max_hp": me.max_hp,
            "energy": me.energy,
            "max_energy": me.max_energy,
            "grounded": True,
            "cubes": me.cubes,
            "upgrades": [],
            "weapons": [
                {
                    "kind": "gun",
                    "cooldown": me.gun_cd,
                    "ammo": me.ammo,
                    "reload": me.reload,
                    "stats": dict(GUN),
                },
                None,
            ],
            "in_zone": math.hypot(me.x - zone["center"][0], me.y - zone["center"][1]) <= zone["radius"],
            "stats": dict(MECH),
        },
        "mechs": mechs,
        "projectiles": prots,
        "crates": [],
        "cubes": [{"id": 5000 + i, "pos": [c.x, c.y, c.z]} for i, c in enumerate(cubes)],
        "pickups": [],
        "sounds": sounds,
        "beams": [],
        "events": events,
        "zone": zone,
        "offer": None,
        "drops": [],
    }


def apply_move(me: Fighter, act, arena, now: float):
    mv = act.get("move") or [0.0, 0.0]
    mx, my = float(mv[0]), float(mv[1])
    ml = math.hypot(mx, my)
    if ml > 1:
        mx /= ml
        my /= ml
    want_boost = bool(act.get("boost")) and me.energy > 4 and ml > 0.05
    speed = BOOST if want_boost else WALK
    dvx = mx * speed - me.vx
    dvy = my * speed - me.vy
    cap = ACCEL * DT
    dm = math.hypot(dvx, dvy)
    if dm > cap and dm > 0:
        dvx *= cap / dm
        dvy *= cap / dm
    me.vx += dvx
    me.vy += dvy
    me.boosting = want_boost
    if want_boost:
        me.energy -= BOOST_DRAIN * DT
        me.last_spend = now
    elif now - me.last_spend > ENERGY_DELAY:
        me.energy = min(me.max_energy, me.energy + ENERGY_REGEN * DT)
    me.energy = max(0.0, me.energy)
    aim = act.get("aim") or [me.aim, me.pitch]
    yaw_step = float(MECH["turn_rate"]) * DT
    pit_step = float(MECH["pitch_rate"]) * DT
    me.aim += max(-yaw_step, min(yaw_step, ang_diff(float(aim[0]), me.aim)))
    me.pitch += max(-pit_step, min(pit_step, float(aim[1]) - me.pitch))
    me.pitch = max(float(MECH["min_pitch"]), min(float(MECH["max_pitch"]), me.pitch))
    nx = me.x + me.vx * DT
    ny = me.y + me.vy * DT
    if not arena.blocked_at(nx, ny):
        me.x, me.y = nx, ny
    elif not arena.blocked_at(nx, me.y):
        me.x = nx
        me.vy *= 0.3
    elif not arena.blocked_at(me.x, ny):
        me.y = ny
        me.vx *= 0.3
    else:
        me.vx = me.vy = 0.0
    me.z = arena.height(me.x, me.y)


def separate(arena, alive):
    for i in range(len(alive)):
        a = alive[i]
        for j in range(i + 1, len(alive)):
            b = alive[j]
            dx = b.x - a.x
            dy = b.y - a.y
            d2 = dx * dx + dy * dy
            if d2 >= 25 or d2 < 1e-6:
                continue
            d = math.sqrt(d2)
            push = (5.0 - d) * 0.45
            ux, uy = dx / d, dy / d
            ax, ay = a.x - ux * push, a.y - uy * push
            bx, by = b.x + ux * push, b.y + uy * push
            if not arena.blocked_at(ax, ay):
                a.x, a.y = ax, ay
            if not arena.blocked_at(bx, by):
                b.x, b.y = bx, by


def fire_gun(me: Fighter, act, bullets, sounds_at):
    if me.gun_cd > 0 or me.reload > 0 or me.ammo <= 0:
        return
    fire = act.get("fire") or [False, False]
    if not fire[0]:
        return
    me.ammo -= 1
    me.gun_cd = GUN_CD
    if me.ammo <= 0:
        me.reload = GUN_RELOAD
    cp = math.cos(me.pitch)
    dx = cp * math.cos(me.aim)
    dy = cp * math.sin(me.aim)
    dz = math.sin(me.pitch)
    bullets.append(
        Bullet(
            me.id,
            me.team,
            me.x + dx * 3.0,
            me.y + dy * 3.0,
            me.z + 7.0,
            dx * GUN_SPEED,
            dy * GUN_SPEED,
            dz * GUN_SPEED,
        )
    )
    sounds_at.append((me, "gun", 320.0))


def step_bullets(arena, bullets, agents, now, measured, pending_events):
    keep = []
    alive = [a for a in agents if a.alive]
    for b in bullets:
        x0, y0, z0 = b.x, b.y, b.z
        dist = GUN_SPEED * DT
        if dist > b.left:
            dist = b.left
        x1 = x0 + b.vx / GUN_SPEED * dist
        y1 = y0 + b.vy / GUN_SPEED * dist
        z1 = z0 + b.vz / GUN_SPEED * dist
        b.left -= dist
        hit = None
        hit_t = 2.0
        steps = max(1, int(dist / 4.0))
        wall_t = 2.0
        for s in range(1, steps + 1):
            tt = s / steps
            if arena.blocked_at(x0 + (x1 - x0) * tt, y0 + (y1 - y0) * tt):
                wall_t = tt
                break
        for a in alive:
            if a.id == b.owner:
                continue
            if a.team == b.team and a.team == 1:
                continue
            dx = a.x - x0
            dy = a.y - y0
            sx = x1 - x0
            sy = y1 - y0
            seg2 = sx * sx + sy * sy
            if seg2 < 1e-6:
                tt = 0.0
            else:
                tt = max(0.0, min(1.0, (dx * sx + dy * sy) / seg2))
            if tt > wall_t:
                continue
            cx = x0 + sx * tt
            cy = y0 + sy * tt
            if (cx - a.x) ** 2 + (cy - a.y) ** 2 > 7.84:
                continue
            zz = z0 + (z1 - z0) * tt
            if zz < a.z - 0.2 or zz > a.z + 9.4:
                continue
            if tt < hit_t:
                hit_t = tt
                hit = a
        if hit is not None:
            dmg = GUN_DMG * (1 + 0.04 * min(10, next(o.cubes for o in agents if o.id == b.owner)))
            bearing = math.atan2(-b.vy, -b.vx)
            rear = abs(ang_diff(bearing, hit.aim)) > HALF_ARC
            if rear:
                dmg *= REAR
            dmg *= max(0.2, 1.0 - hit.armor)
            hit.hp -= dmg
            hit.last_damage = now
            if hit.hp <= 0 and hit.cause is None:
                hit.cause = "gun"
            if hit is measured:
                pending_events.append(
                    {
                        "type": "damaged",
                        "amount": dmg,
                        "weapon": "gun",
                        "bearing": bearing,
                        "by": b.owner,
                        "rear": rear,
                    }
                )
            continue
        if wall_t <= 1.0 or b.left <= 0:
            continue
        b.x, b.y, b.z = x1, y1, z1
        keep.append(b)
    bullets[:] = keep


def give_cube(a: Fighter):
    old = a.max_hp
    a.cubes += 1
    a.max_hp = old * (1 + float(CFG["cubes"]["hp_per_cube"]))
    a.hp = min(a.max_hp, a.hp + (a.max_hp - old))


def resolve_deaths(agents, cubes, now):
    dying = [a for a in agents if a.alive and a.hp <= 0]
    if not dying:
        return
    place = sum(1 for a in agents if a.alive)
    for a in dying:
        a.alive = False
        a.place = place
        a.death_t = now
        a.hp = 0
        cubes.append(CubePile(a.x, a.y, a.z, a.cubes + int(CFG["cubes"]["death_drop_extra"])))


def finish_timeout(agents, now):
    survivors = [a for a in agents if a.alive]
    survivors.sort(key=lambda a: (-a.hp, a.id))
    i = 0
    while i < len(survivors):
        j = i + 1
        while j < len(survivors) and survivors[j].hp == survivors[i].hp:
            j += 1
        for k in range(i, j):
            survivors[k].place = i + 1
            survivors[k].death_t = now
            survivors[k].cause = "time"
        i = j


def run_match(seed: int, mode: str, arena: nav.Arena):
    rng = random.Random(seed)
    zone_model = Zone(rng)
    pts = spawn_points(rng, arena)
    rng.shuffle(pts)
    kinds = []
    if mode == "ffa":
        kinds = ["shade"] + ["baseline"] * 15
    elif mode == "squad":
        kinds = ["shade"] + ["squad"] * 15
    elif mode == "baseline-ffa":
        kinds = ["baseline"] * 16
    elif mode == "baseline-squad":
        kinds = ["baseline"] + ["squad"] * 15
    else:
        raise SystemExit(f"unknown mode {mode}")
    agents = []
    for i, kind in enumerate(kinds):
        x, y = pts[i]
        agents.append(Fighter(i, x, y, arena.height(x, y), kind))
    measured = agents[0]
    if measured.kind == "shade":
        hello = {
            "type": "hello",
            "protocol": "mechfight/1",
            "slot": 0,
            "seed": seed,
            "map": {"dir": ARENA_DIR, "files": ["v2/world.json", "v2/terrain.i16"]},
            "config": CFG,
        }
        measured.bot = Bot(hello)
    cubes = []
    for _ in range(28):
        for _try in range(40):
            ang = rng.random() * math.tau
            rad = 500 * math.sqrt(rng.random())
            x = 41.3 + math.cos(ang) * rad
            y = -72.8 + math.sin(ang) * rad
            if not arena.blocked_at(x, y):
                cubes.append(CubePile(x, y, arena.height(x, y), 1))
                break
    bullets = []
    shared = {"orbit": rng.random() * math.tau, "t": 0.0}
    pending_sounds = []
    pending_events = []
    tick = 0
    now = 0.0
    errors = 0
    while now < DURATION and sum(1 for a in agents if a.alive) > 1 and measured.alive:
        zone = zone_model.at(now)
        alive = [a for a in agents if a.alive]
        views = {a.id: visible_ids(arena, agents, a, now) for a in alive}
        actions = {}
        shared["t"] = now
        for a in alive:
            enemies = views[a.id]
            if a.kind == "shade":
                obs = make_obs(a, enemies, zone, now, tick, bullets, pending_sounds, pending_events, cubes)
                obs["alive"] = len(alive)
                try:
                    actions[a.id] = a.bot.act(obs) or {}
                except Exception:
                    errors += 1
                    if errors <= 3:
                        traceback.print_exc()
                    actions[a.id] = {"move": [0, 0], "aim": [a.aim, 0]}
            elif a.kind == "squad":
                actions[a.id] = squad_action(a, enemies, zone, shared, arena, tick)
            else:
                actions[a.id] = baseline_action(a, enemies, zone, tick)
        pending_sounds = []
        pending_events = []
        # Footsteps from the previous velocity, heard by the measured bot.
        for a in alive:
            if a is measured or not measured.alive:
                continue
            d = math.hypot(a.x - measured.x, a.y - measured.y)
            spd = math.hypot(a.vx, a.vy)
            if a.boosting and d < 90:
                pending_sounds.append(
                    {
                        "kind": "boost",
                        "bearing": math.atan2(a.y - measured.y, a.x - measured.x),
                        "dist": "near" if d < 60 else "mid",
                        "age": 0.05,
                    }
                )
            elif spd > 5 and d < 40 and now >= a.next_step:
                a.next_step = now + 0.6
                pending_sounds.append(
                    {
                        "kind": "step",
                        "bearing": math.atan2(a.y - measured.y, a.x - measured.x),
                        "dist": "near",
                        "age": 0.05,
                    }
                )
        fired = []
        for a in alive:
            apply_move(a, actions[a.id], arena, now)
            if a.gun_cd > 0:
                a.gun_cd = max(0.0, a.gun_cd - DT)
            if a.reload > 0:
                a.reload = max(0.0, a.reload - DT)
                if a.reload == 0:
                    a.ammo = GUN_MAG
            before = len(bullets)
            fire_gun(a, actions[a.id], bullets, fired)
            if len(bullets) > before and measured.alive and a is not measured:
                d = math.hypot(a.x - measured.x, a.y - measured.y)
                if d < 320:
                    pending_sounds.append(
                        {
                            "kind": "gun",
                            "bearing": math.atan2(a.y - measured.y, a.x - measured.x),
                            "dist": "near" if d < 60 else "mid" if d < 160 else "far",
                            "age": 0.05,
                        }
                    )
        separate(arena, [a for a in agents if a.alive])
        step_bullets(arena, bullets, agents, now, measured, pending_events)
        zc = zone["center"]
        for a in agents:
            if not a.alive:
                continue
            outside = math.hypot(a.x - zc[0], a.y - zc[1]) > zone["radius"]
            if outside:
                a.hp -= zone["dps"] * DT
                a.last_damage = now
                if a.hp <= 0 and a.cause is None:
                    a.cause = "zone"
                if a is measured and a.hp > 0:
                    pending_events.append(
                        {"type": "damaged", "amount": zone["dps"] * DT, "weapon": "zone", "bearing": None, "by": None}
                    )
            elif now - a.last_damage > REGEN_DELAY and a.hp < a.max_hp:
                a.hp = min(a.max_hp, a.hp + REGEN * DT)
        # Cube pickup: closest mech inside 3.5 m, lower id wins ties.
        remain = []
        for pile in cubes:
            closest = None
            best = 1e9
            for a in agents:
                if not a.alive:
                    continue
                d = math.hypot(a.x - pile.x, a.y - pile.y)
                key = d + a.id * 1e-4
                if d <= float(CFG["cubes"]["pickup_radius"]) and key < best:
                    best = key
                    closest = a
            if closest is None:
                remain.append(pile)
                continue
            for _ in range(pile.n):
                give_cube(closest)
        cubes = remain
        resolve_deaths(agents, cubes, now)
        shared["orbit"] += 0.035 * DT
        now += DT
        tick += 2  # server decide_every is 2 sim ticks; echo values only need to be unique
        if measured.place is not None:
            break
    if measured.alive and sum(1 for a in agents if a.alive) == 1:
        measured.place = 1
        measured.cause = "win"
        measured.death_t = now
    elif measured.alive:
        finish_timeout(agents, now)
    return {
        "seed": seed,
        "place": measured.place,
        "score": 16 - measured.place,
        "time": measured.death_t if measured.death_t is not None else now,
        "cause": measured.cause,
        "hp": round(measured.hp, 1),
        "cubes": measured.cubes,
        "errors": errors,
        "alive": sum(1 for a in agents if a.alive),
    }


def summarize(mode: str, rows):
    n = len(rows)
    mean_score = sum(r["score"] for r in rows) / n
    mean_place = sum(r["place"] for r in rows) / n
    mean_t = sum(r["time"] for r in rows) / n
    causes = {}
    for r in rows:
        causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    print(
        f"{mode}: matches={n} mean_score={mean_score:.3f} mean_place={mean_place:.3f} "
        f"mean_survival_s={mean_t:.1f} causes={causes}"
    )
    print("  places:", " ".join(str(r["place"]) for r in rows))
    return {"mode": mode, "matches": n, "mean_score": mean_score, "mean_place": mean_place, "mean_survival_s": mean_t, "causes": causes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ffa", choices=["ffa", "squad", "baseline-ffa", "baseline-squad", "all"])
    ap.add_argument("--matches", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.perf_counter()
    arena = nav.get_arena(ARENA_DIR)
    print(f"arena loaded in {time.perf_counter() - t0:.2f}s ok={arena.ok}")
    modes = ["ffa", "squad", "baseline-ffa", "baseline-squad"] if args.mode == "all" else [args.mode]
    for mode in modes:
        rows = []
        mt = time.perf_counter()
        for i in range(args.matches):
            seed = args.seed0 + i
            row = run_match(seed, mode, arena)
            rows.append(row)
            print(
                f"  {mode} seed {seed}: place {row['place']} score {row['score']} "
                f"t={row['time']:.1f}s cause={row['cause']} hp={row['hp']} cubes={row['cubes']} err={row['errors']}"
            )
        summarize(mode, rows)
        print(f"  elapsed {time.perf_counter() - mt:.1f}s")


if __name__ == "__main__":
    main()
