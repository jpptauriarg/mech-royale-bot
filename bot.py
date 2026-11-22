"""Oval Ghost: live longer than the hunters.

Practice matches are one squad of house bots that hunt only us, so the score
is how many of them die before we do. Stay in the next circle, on the hidden
side of a building, and only take a fight when a single mech can see us.
"""
import math
import os
import random

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

import mechfight as mf
from world import World

HERE = os.path.dirname(os.path.abspath(__file__))
TAU = 2 * math.pi

# Durability first: hit points, then armor, then regen. Speed after that.
UPGRADES = [
    "plating", "armor", "nanites", "servos", "jets", "reactor", "capacitor",
    "gyros", "sensors", "gun_mag", "gun_damage", "gun_rate", "gun_velocity",
    "laser_damage", "laser_charge", "laser_pierce", "lightning_efficiency",
    "lightning_range", "lightning_chain", "mortar_reload", "mortar_velocity",
    "mortar_splash",
]
WEAPON_SCORE = {"laser": 3, "lightning": 2, "gun": 0, "mortar": -1}


def angdiff(a, b):
    return (a - b + math.pi) % TAU - math.pi


class Bot:
    name = "oval-ghost"

    def __init__(self, hello):
        cfg = hello["config"]
        mech = cfg["mech"]
        self.min_pitch = float(mech["min_pitch"])
        self.max_pitch = float(mech["max_pitch"])
        self.walk = float(mech["walk_speed"])
        self.boost_spd = float(mech["boost_speed"])
        self.gun_speed = float(cfg["weapons"]["gun"]["speed"])
        self.pickup_r = float(cfg["loot"]["pickup_radius"])
        self.npc_aggro = float(cfg["npc"]["aggro_range"])
        self.rng = random.Random(int(hello.get("seed", 1)))
        self.W = World(os.path.join(HERE, "maps.npz"))
        self._build_graph()
        self.goal = None
        self.path = []
        self.plan_t = -99.0
        self.plan_key = None
        self.attackers = {}
        self.bearings = []
        self.stuck = 0.0
        self.prev = None
        self.sidestep = None
        self.strafe = 1
        self.strafe_until = 0.0
        self.mode = "hide"
        self.scan = self.rng.random() * TAU
        mf.log("oval-ghost nav", int(self.main.sum()), "free cells")

    def _build_graph(self):
        nb = self.W.navblk
        surf = self.W.navsurf
        h, w = nb.shape
        self.H, self.Wd = h, w
        self.cell = self.W.cell
        idx = np.arange(h * w, dtype=np.int32).reshape(h, w)
        wall_dist = distance_transform_edt(~nb)
        self.wall_dist = wall_dist.ravel().astype(np.float32)
        rows, cols, weights = [], [], []
        for dj, di, steps in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
            j0, j1 = 0, h - dj
            i0, i1 = max(0, -di), w - max(0, di)
            a = idx[j0:j1, i0:i1]
            b = idx[j0 + dj:j1 + dj, i0 + di:i1 + di]
            ok = (
                ~nb[j0:j1, i0:i1]
                & ~nb[j0 + dj:j1 + dj, i0 + di:i1 + di]
                & (np.abs(surf[j0:j1, i0:i1] - surf[j0 + dj:j1 + dj, i0 + di:i1 + di]) < 2.4 * steps)
            )
            aa, bb = a[ok], b[ok]
            open_cost = 0.5 * (np.minimum(self.wall_dist[aa], 8) + np.minimum(self.wall_dist[bb], 8))
            weight = (self.cell * steps) * (1.0 + 0.22 * open_cost)
            rows.append(aa)
            cols.append(bb)
            weights.append(weight.astype(np.float32))
        src = np.concatenate(rows + cols)
        dst = np.concatenate(cols + rows)
        wgt = np.concatenate(weights + weights)
        self.graph = csr_matrix((wgt, (src, dst)), shape=(h * w, h * w))
        jj, ii = np.mgrid[0:h, 0:w]
        self.cx = (self.W.x0 + self.cell * ii + self.cell * 0.5).ravel()
        self.cy = (self.W.y0 + self.cell * jj + self.cell * 0.5).ravel()
        self.blk = nb.ravel()
        _, labels = connected_components(self.graph, directed=False)
        labels = labels.copy()
        labels[self.blk] = -1
        counts = np.bincount(labels[labels >= 0])
        big = int(counts.argmax()) if counts.size else 0
        self.main = labels == big
        _, nearest = distance_transform_edt(~self.main.reshape(h, w), return_indices=True)
        self.near_main = (nearest[0] * w + nearest[1]).ravel()

    def node(self, x, y):
        i = int((x - self.W.x0) // self.cell)
        j = int((y - self.W.y0) // self.cell)
        i = min(max(i, 0), self.Wd - 1)
        j = min(max(j, 0), self.H - 1)
        n = j * self.Wd + i
        if not self.blk[n]:
            return n
        best = None
        for radius in range(1, 6):
            for dj in range(-radius, radius + 1):
                for di in range(-radius, radius + 1):
                    jj, ii = j + dj, i + di
                    if 0 <= jj < self.H and 0 <= ii < self.Wd and not self.blk[jj * self.Wd + ii]:
                        d = dj * dj + di * di
                        if best is None or d < best[0]:
                            best = (d, jj * self.Wd + ii)
            if best:
                return best[1]
        return n

    def free_at(self, x, y):
        i = int((x - self.W.x0) // self.cell)
        j = int((y - self.W.y0) // self.cell)
        if not (0 <= i < self.Wd and 0 <= j < self.H):
            return False
        return not self.blk[j * self.Wd + i]

    def safe_dir(self, x, y, yaw, look=7.0):
        for off in (0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.3, -2.3, math.pi):
            a = yaw + off
            if self.free_at(x + look * math.cos(a), y + look * math.sin(a)):
                return a
        return yaw

    def act(self, obs):
        me = obs["self"]
        x, y, z = (float(v) for v in me["pos"])
        t = float(obs.get("t", 0.0))
        stats = me.get("stats") or {}
        walk = float(stats.get("walk_speed", self.walk))
        zone = obs.get("zone") or {}
        hpf = float(me["hp"]) / max(float(me.get("max_hp") or 1), 1.0)
        act = {}

        offer = obs.get("offer")
        if offer and offer.get("choices"):
            act["pick"] = self._pick(offer["choices"])

        for ev in obs.get("events") or []:
            if ev.get("type") == "damaged" and ev.get("by") is not None:
                self.attackers[ev["by"]] = t
                if ev.get("bearing") is not None and ev.get("weapon") != "zone":
                    self.bearings.append((t, float(ev["bearing"])))
        self.bearings = [(tt, b) for tt, b in self.bearings if t - tt < 3.0]
        recent = {i for i, tt in self.attackers.items() if t - tt < 4.0}

        enemies = []
        npcs = []
        for actor in obs.get("mechs") or []:
            if actor.get("kind") == "npc":
                npcs.append(actor)
            elif actor.get("kind") == "mech":
                enemies.append(actor)
        eye_z = z + float(stats.get("eye_height", 8.0))
        muzzle_z = z + float(stats.get("muzzle_height", 7.0))
        if enemies:
            feet = np.array([[e["pos"][0], e["pos"][1], e["pos"][2]] for e in enemies], np.float64)
            vis = self.W.sees(np.tile([x, y, eye_z], (len(enemies), 1)), feet, canopy=False)
        else:
            vis = []
        for e, seen in zip(enemies, vis):
            e["_d"] = math.hypot(float(e["pos"][0]) - x, float(e["pos"][1]) - y)
            e["_los"] = bool(seen)
            bearing = math.atan2(y - float(e["pos"][1]), x - float(e["pos"][0]))
            aim = e.get("aim") or [0.0, 0.0]
            e["_facing"] = abs(angdiff(float(aim[0]), bearing)) < 0.45 or bool(e.get("charging"))
        enemies.sort(key=lambda e: e["_d"])
        visible = [e for e in enemies if e["_los"] and e["_d"] < 220.0]
        hostile = [
            e for e in visible
            if e["_d"] < 140.0 or e["_facing"] or e["id"] in recent
        ]
        radius = float(zone.get("radius", 9999.0))
        endgame = radius < 75.0

        if hostile:
            weakest = min(hostile, key=lambda e: float(e.get("hp", 1400)))
            weaker = float(weakest.get("hp", 1400)) < float(me["hp"])
            close = weakest["_d"] < 95.0
            # A long sightline against a healthy hunter is a retreat back into cover,
            # not a duel. Take the fight only when it is one mech and it is close.
            lone_close = len(hostile) == 1 and close and (hpf >= 0.4 or weaker or float(weakest.get("hp", 1400)) < 380)
            if lone_close or (endgame and len(hostile) <= 2 and hpf >= 0.38):
                self.mode = "fight"
            else:
                self.mode = "retreat"
        elif recent and t - max(self.attackers.values()) < 2.5:
            self.mode = "retreat"
        else:
            self.mode = "hide"

        heard = self._heard(obs.get("sounds") or [])
        if heard and self.mode == "hide":
            # A gunshot or a boost we cannot see yet is a hunter closing in.
            self.mode = "retreat"
            self.bearings.append((t, heard[0]))

        threats = list(visible[:4])
        if heard:
            for bearing, dist in heard[1]:
                threats.append({
                    "id": -1,
                    "pos": [x + math.cos(bearing) * dist, y + math.sin(bearing) * dist, z],
                    "_d": dist,
                    "_los": False,
                })
        for _, bearing in self.bearings[-2:]:
            threats.append({
                "id": -1,
                "pos": [x + 75.0 * math.cos(bearing), y + 75.0 * math.sin(bearing), z],
                "_d": 75.0,
                "_los": False,
            })
        for npc in npcs:
            d = math.hypot(float(npc["pos"][0]) - x, float(npc["pos"][1]) - y)
            if d < self.npc_aggro + 8:
                threats.append(npc)

        key = (self.mode, tuple(sorted(int(e.get("id", -1)) for e in visible[:4])))
        replan_every = 1.0 if self.mode == "retreat" else 2.2
        if t - self.plan_t > replan_every or not self.path or key != self.plan_key:
            try:
                self._plan(x, y, z, zone, threats if self.mode != "fight" else threats)
                self.plan_t = t
                self.plan_key = key
            except Exception as exc:
                mf.log("plan failed", repr(exc))
                self.path = []

        cx, cy = _xy(zone.get("center"), (x, y))
        dist_c = math.hypot(x - cx, y - cy)
        in_zone = bool(me.get("in_zone", True))
        zone_urgent = (not in_zone) or dist_c > float(zone.get("radius", 1e9)) - 8.0
        cur = self.node(x, y)
        pocket = not bool(self.main[cur])

        move_yaw = None
        speed = 1.0
        boost = False
        energy = float(me.get("energy", 0))
        if pocket:
            n = int(self.near_main[cur])
            move_yaw = math.atan2(float(self.cy[n]) - y, float(self.cx[n]) - x)
            boost = energy > 4
            if me.get("grounded", True):
                act["jump"] = True
        else:
            move_yaw = self._steer(x, y)
            if move_yaw is None:
                speed = 0.0

        target = None
        if hostile:
            def rank(e):
                score = e["_d"]
                if e["id"] in recent:
                    score -= 70
                score -= (1400 - float(e.get("hp", 1400))) * 0.1
                if e.get("charging"):
                    score -= 40
                return score
            target = min(hostile, key=rank)
        elif visible:
            target = visible[0]

        if self.mode == "fight" and target is not None and not zone_urgent and not pocket:
            bearing = math.atan2(float(target["pos"][1]) - y, float(target["pos"][0]) - x)
            if t > self.strafe_until:
                if self.rng.random() < 0.65:
                    self.strafe = -self.strafe
                self.strafe_until = t + self.rng.uniform(0.4, 0.85)
            lo, hi = 55.0, 130.0
            mine = {w["kind"] for w in (me.get("weapons") or []) if w}
            theirs = set(target.get("weapons") or [])
            if "lightning" in mine and "lightning" not in theirs and float(target.get("hp", 1400)) <= float(me["hp"]):
                lo, hi = 16.0, 38.0
            elif "lightning" in theirs:
                lo, hi = 62.0, 150.0
            radial = 0.0
            if target["_d"] < lo:
                radial = -0.85
            elif target["_d"] > hi:
                radial = 0.75
            sx = math.cos(bearing + math.pi / 2 * self.strafe)
            sy = math.sin(bearing + math.pi / 2 * self.strafe)
            mvx = sx + radial * math.cos(bearing)
            mvy = sy + radial * math.sin(bearing)
            if dist_c > float(zone.get("radius", 1e9)) - 22.0:
                inward = math.atan2(cy - y, cx - x)
                mvx += math.cos(inward)
                mvy += math.sin(inward)
            move_yaw = math.atan2(mvy, mvx)
            speed = 1.0
            boost = energy > 45 and target["_d"] < 70
        elif self.mode == "retreat":
            boost = energy > 12

        shrink_end = float(zone.get("shrink_end", t + 999))
        time_left = max(0.0, shrink_end - t) if in_zone else 0.0
        path_m = len(self.path) * self.cell
        if zone_urgent or (path_m > walk * max(time_left - 6.0, 0.4) and path_m > 18):
            boost = energy > 6 or not in_zone
            speed = 1.0
            if move_yaw is None:
                nx, ny = _xy(zone.get("next_center"), (cx, cy))
                move_yaw = math.atan2(ny - y, nx - x)

        weapons = list(me.get("weapons") or [])
        while len(weapons) < 2:
            weapons.append(None)
        aim_yaw, aim_pitch, fire = self._aim_and_fire(
            me, x, y, muzzle_z, target, weapons, visible, hostile, energy
        )
        if aim_yaw is None:
            if self.bearings:
                aim_yaw = self.bearings[-1][1]
            elif move_yaw is not None and speed > 0:
                aim_yaw = move_yaw
            else:
                self.scan += 0.35
                aim_yaw = self.scan
            aim_pitch = 0.02

        dodge = self._dodge(x, y, z, obs.get("projectiles") or [], visible)
        # Only an imminent shot may change heading. A stream of bullets used to
        # replace the path to cover, so the mech danced in the open and died.
        if dodge is not None and not pocket and dodge[0] < 0.32:
            _tca, dodge_yaw = dodge
            if move_yaw is None:
                move_yaw = dodge_yaw
            else:
                px, py = math.cos(move_yaw), math.sin(move_yaw)
                dx, dy = math.cos(dodge_yaw), math.sin(dodge_yaw)
                move_yaw = math.atan2(py + dy, px + dx)
            speed = 1.0
            if self.mode == "retreat":
                boost = energy > 8
            elif energy > 22:
                boost = True

        if self.prev is not None and move_yaw is not None and speed > 0.5 and not pocket:
            moved = math.hypot(x - self.prev[0], y - self.prev[1])
            self.stuck = self.stuck + 0.1 if moved < 0.25 else max(0.0, self.stuck - 0.2)
        self.prev = (x, y)
        if self.stuck > 0.7 and me.get("grounded", True):
            act["jump"] = True
        if self.stuck > 1.4 and (self.sidestep is None or t > self.sidestep[1]):
            self.sidestep = (self.rng.random() * TAU, t + 0.6)
        if self.stuck > 3.0:
            self.path = []
            self.stuck = 0.0
        if self.sidestep and t < self.sidestep[1]:
            move_yaw = self.sidestep[0]
            speed = 1.0

        if move_yaw is not None and speed > 0:
            move_yaw = self.safe_dir(x, y, move_yaw)
            act["move"] = [round(math.cos(move_yaw) * speed, 4), round(math.sin(move_yaw) * speed, 4)]
        else:
            act["move"] = [0.0, 0.0]
            boost = False
        act["boost"] = bool(boost and (speed > 0 or not me.get("grounded", True)))
        act["aim"] = [
            round(((aim_yaw + math.pi) % TAU) - math.pi, 4),
            round(max(self.min_pitch, min(self.max_pitch, aim_pitch)), 4),
        ]
        act["fire"] = fire
        self._loot(obs, me, act, weapons)
        if self.mode != "fight" and not visible:
            gun = weapons[0]
            if gun and gun.get("kind") == "gun" and float(gun.get("reload") or 0) <= 0:
                ammo = float(gun.get("ammo", 0))
                mag = float((gun.get("stats") or {}).get("mag", 12))
                if ammo <= max(2.0, mag * 0.34):
                    act["reload"] = True
        return act

    def _plan(self, x, y, z, zone, threats):
        src = self.node(x, y)
        dist, pred = dijkstra(self.graph, indices=src, directed=False, return_predecessors=True, limit=900.0)
        reach = np.isfinite(dist) & ~self.blk & self.main
        nx, ny = _xy(zone.get("next_center"), _xy(zone.get("center"), (x, y)))
        nr = float(zone.get("next_radius", 0.0))
        ccx, ccy = _xy(zone.get("center"), (x, y))
        cr = float(zone.get("radius", 1e9))
        dn = np.hypot(self.cx - nx, self.cy - ny)
        if nr < 12.0:
            ok = reach
            score_all = dist + 1.6 * dn
        else:
            margin = max(8.0, min(24.0, nr * 0.08))
            rel = dn / max(nr - margin, 1.0)
            ok = reach & (dn < max(nr - margin, 6.0))
            if int(ok.sum()) < 12:
                ok = reach & (dn < nr + 10.0)
            if int(ok.sum()) < 5:
                ok = reach
            score_all = (
                0.10 * dist
                + 18.0 * np.maximum(rel - 0.45, 0.0) ** 2
                + 5.5 * np.minimum(self.wall_dist, 12.0)
            )
        dc = np.hypot(self.cx - ccx, self.cy - ccy)
        score_all = score_all + np.where(dc > cr - 6.0, 350.0, 0.0)
        sel = np.flatnonzero(ok)
        if sel.size == 0:
            self.goal = src
            self.path = []
            return
        score = score_all[sel]
        if self.goal is not None and 0 <= self.goal < score_all.size:
            score = score - np.where(sel == self.goal, 12.0, 0.0)
        order = np.argsort(score)
        if threats:
            # Closest cells first. A long run to a prettier corner is how gunfire catches us.
            near = np.argsort(dist[sel])[:400]
            cand = sel[near]
            sc = (0.4 * dist[cand]).astype(np.float64)
            feet_z = self.W.surf(self.cx[cand], self.cy[cand])
            feet = np.stack([self.cx[cand], self.cy[cand], np.asarray(feet_z, np.float64)], 1)
            for e in threats[:4]:
                ex, ey, ez = (float(v) for v in e["pos"])
                eye = np.tile([ex, ey, ez + 8.0], (len(cand), 1))
                seen = self.W.sees(eye, feet, canopy=True)
                weight = 250.0 if int(e.get("id", 0)) >= 0 else 120.0
                sc += seen.astype(np.float64) * weight
            hidden = np.flatnonzero(sc < 80.0)
            if hidden.size:
                best = int(cand[hidden[int(np.argmin(dist[cand[hidden]]))]])
            else:
                best = int(cand[int(np.argmin(sc))])
        else:
            best = int(sel[int(order[0])])
        self.goal = best
        path = []
        n = best
        guard = 0
        while n != src and n >= 0 and guard < 4000:
            path.append(int(n))
            nxt = int(pred[n])
            if nxt < 0:
                path = []
                break
            n = nxt
            guard += 1
        else:
            path.reverse()
        if n != src:
            path = []
        self.path = path

    def _steer(self, x, y):
        while self.path and math.hypot(float(self.cx[self.path[0]]) - x, float(self.cy[self.path[0]]) - y) < self.cell * 0.75:
            self.path.pop(0)
        if not self.path:
            if self.goal is None:
                return None
            if math.hypot(float(self.cx[self.goal]) - x, float(self.cy[self.goal]) - y) < self.cell:
                return None
            return math.atan2(float(self.cy[self.goal]) - y, float(self.cx[self.goal]) - x)
        n = self.path[min(2, len(self.path) - 1)]
        return math.atan2(float(self.cy[n]) - y, float(self.cx[n]) - x)

    def _aim_and_fire(self, me, x, y, mz, target, weapons, visible, hostile, energy):
        fire = [False, False]
        if target is None:
            for i, wp in enumerate(weapons[:2]):
                if wp and wp.get("kind") == "laser" and float(wp.get("charge") or 0) > 0.05:
                    fire[i] = False
            return None, 0.0, fire
        hitscan = False
        for wp in weapons:
            if wp and wp.get("kind") == "laser" and float(wp.get("charge") or 0) > float((wp.get("stats") or {}).get("charge", 0.75)) * 0.55:
                hitscan = True
        speed = self.gun_speed
        for wp in weapons:
            if wp and wp.get("kind") == "gun":
                speed = float((wp.get("stats") or {}).get("speed", speed))
        yaw, pitch = self._lead(x, y, mz, target, speed, hitscan)
        err = abs(angdiff(float(me["aim"][0]), yaw))
        dist = max(target["_d"], 1.0)
        tol = math.atan2(2.4, dist) + 0.01
        shoot = self.mode == "fight" or target in hostile
        los = bool(self.W.los(
            [x, y, mz],
            [float(target["pos"][0]), float(target["pos"][1]), float(target["pos"][2]) + 4.5],
            canopy=False,
        )[0])
        for i, wp in enumerate(weapons[:2]):
            if not wp:
                continue
            kind = wp.get("kind")
            wst = wp.get("stats") or {}
            if not shoot and not (kind == "laser" and float(wp.get("charge") or 0) > 0.05):
                continue
            if kind == "gun":
                rng = float(wst.get("range", 230))
                if shoot and los and err < tol and float(wp.get("reload") or 0) <= 0 and float(wp.get("ammo") or 0) > 0 and dist < rng * 0.97:
                    fire[i] = True
            elif kind == "laser":
                full = float(wst.get("charge", 0.75))
                held = float(wp.get("charge") or 0)
                if not shoot:
                    fire[i] = False
                    continue
                if dist > float(wst.get("range", 280)):
                    fire[i] = False
                    continue
                if held <= 0.02:
                    fire[i] = bool(shoot and float(wp.get("cooldown") or 0) <= 0.05 and dist > 18)
                elif held < full * 0.94:
                    fire[i] = True
                else:
                    fire[i] = not (err < tol and los)
            elif kind == "lightning":
                rng = float(wst.get("range", 45))
                cone = math.radians(float(wst.get("cone_deg", 30)))
                if shoot and dist < rng * 0.95 and err < cone * 0.45 and energy > 10:
                    fire[i] = True
        return yaw, pitch, fire

    def _lead(self, x, y, mz, enemy, speed, hitscan):
        tx, ty, tz = (float(v) for v in enemy["pos"])
        vel = enemy.get("vel") or [0.0, 0.0, 0.0]
        vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
        flight = 0.0
        if not hitscan:
            for _ in range(3):
                px = tx + vx * flight
                py = ty + vy * flight
                flight = math.hypot(px - x, py - y) / max(speed, 1.0)
        px = tx + vx * flight
        py = ty + vy * flight
        pz = tz + 4.5 + vz * flight
        horiz = math.hypot(px - x, py - y) or 1.0
        return math.atan2(py - y, px - x), math.atan2(pz - mz, horiz)

    def _heard(self, sounds):
        """Bearings worth running from. Returns (primary bearing or None, threat list)."""
        rank = {"near": 0, "mid": 1, "far": 2}
        interesting = []
        for s in sounds:
            kind = s.get("kind")
            if kind not in ("gun", "boost", "step", "laser", "laser_charge", "lightning", "mortar"):
                continue
            if kind in ("step", "boost") and s.get("dist") == "far":
                continue
            dist = {"near": 45.0, "mid": 110.0, "far": 200.0}.get(s.get("dist"), 120.0)
            if kind == "gun" and s.get("dist") == "far":
                dist = 200.0
            interesting.append((rank.get(s.get("dist"), 2), 0 if kind == "gun" else 1, float(s.get("bearing", 0.0)), dist))
        if not interesting:
            return None
        interesting.sort()
        threats = [(b, d) for _r, _k, b, d in interesting[:3]]
        return interesting[0][2], threats

    def _dodge(self, x, y, z, projectiles, visible):
        """Return (seconds-to-impact, sidestep yaw) for the most urgent shot."""
        best = None
        for e in visible:
            if not e.get("charging") or not e.get("_facing"):
                continue
            away = math.atan2(y - float(e["pos"][1]), x - float(e["pos"][0]))
            if abs(angdiff(float((e.get("aim") or [0, 0])[0]), away)) < 0.14:
                yaw = away + math.pi / 2 * self.strafe
                if best is None or 0.15 < best[0]:
                    best = (0.15, yaw)
        mz = z + 4.5
        for p in projectiles:
            pos = p.get("pos") or [0, 0, 0]
            vel = p.get("vel") or [0, 0, 0]
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
            v2 = vx * vx + vy * vy + vz * vz
            if v2 < 1.0:
                continue
            tca = ((x - px) * vx + (y - py) * vy + (mz - pz) * vz) / v2
            if tca < 0.05 or tca > 1.15:
                continue
            miss = math.hypot(x - (px + vx * tca), y - (py + vy * tca), mz - (pz + vz * tca))
            limit = 9.0 if p.get("kind") == "mortar" else 4.2
            if miss < limit and (best is None or tca < best[0]):
                side = 1.0 if ((x - px) * -vy + (y - py) * vx) > 0 else -1.0
                best = (tca, math.atan2(vy, vx) + math.pi / 2 * side)
        return best

    def _loot(self, obs, me, act, weapons):
        x, y, z = (float(v) for v in me["pos"])
        held = {w.get("kind") for w in weapons if w}
        best = None
        best_s = -1
        for pickup in obs.get("pickups") or []:
            pos = pickup.get("pos") or [0, 0, 0]
            if abs(float(pos[2]) - z) > 3.8:
                continue
            if math.hypot(float(pos[0]) - x, float(pos[1]) - y) > self.pickup_r - 0.2:
                continue
            kind = pickup.get("kind")
            score = WEAPON_SCORE.get(kind, -1)
            if kind in held or score < 1:
                continue
            if score > best_s:
                best_s = score
                best = pickup
        if best is None:
            return
        if weapons[1] is None:
            act["take"] = best["id"]
            return
        current = weapons[1]
        cur = WEAPON_SCORE.get(current.get("kind"), 0)
        charging = current.get("kind") == "laser" and float(current.get("charge") or 0) > 0.05
        if best_s > cur and not charging:
            act["take"] = best["id"]
            act["drop"] = 1

    def _pick(self, choices):
        best_i, best_r = 0, 10 ** 9
        for i, choice in enumerate(choices):
            ident = choice.get("id")
            rank = UPGRADES.index(ident) if ident in UPGRADES else 50
            if rank < best_r:
                best_r = rank
                best_i = i
        return best_i


def _xy(value, default):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return default


if __name__ == "__main__":
    mf.run(Bot)
