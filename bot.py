"""Oval Ghost: survive the zone, avoid fights, answer every tick.

Placement is the score. The mech walks into the next circle early enough to
arrive before the ring, boosts only when it would be caught, and keeps away
from other mechs and NPC drones. It stays under step-sound speed while it is
already safe, and it fires only at a lone close threat or in the final circles.
"""
import math

import mechfight as mf

# Lower is better. Weapon upgrades sit below survival stats; unknown ids are last.
UPGRADE_RANK = {
    "nanites": 0,
    "armor": 1,
    "plating": 2,
    "servos": 3,
    "jets": 4,
    "reactor": 5,
    "capacitor": 6,
    "sensors": 7,
    "gyros": 8,
    "gun_mag": 20,
    "gun_damage": 21,
    "gun_rate": 22,
    "gun_velocity": 23,
    "laser_damage": 24,
    "laser_charge": 25,
    "laser_pierce": 26,
    "mortar_reload": 27,
    "mortar_splash": 28,
    "mortar_velocity": 29,
    "lightning_range": 30,
    "lightning_efficiency": 31,
    "lightning_chain": 32,
}
LOOT_RANK = {"laser": 0, "mortar": 1, "lightning": 2, "gun": 3}


def _hypot(x, y):
    return math.hypot(x, y)


def _angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _clamp_disk(px, py, cx, cy, radius):
    dx, dy = px - cx, py - cy
    d = _hypot(dx, dy)
    if radius <= 0:
        return cx, cy
    if d > radius and d > 1e-6:
        k = radius / d
        return cx + dx * k, cy + dy * k
    return px, py


class Bot:
    name = "oval-ghost"

    def __init__(self, hello):
        cfg = hello["config"]
        mech = cfg["mech"]
        weapons = cfg["weapons"]
        self.walk = float(mech["walk_speed"])
        self.boost_spd = float(mech["boost_speed"])
        self.min_pitch = float(mech["min_pitch"])
        self.max_pitch = float(mech["max_pitch"])
        self.step_speed = float(cfg["sound"]["step_speed"])
        self.pickup_r = float(cfg["loot"]["pickup_radius"])
        self.npc_aggro = float(cfg["npc"]["aggro_range"])
        self.gun_speed = float(weapons["gun"]["speed"])
        self.gun_range = float(weapons["gun"]["range"])
        self.mortar_speed = float(weapons["mortar"]["speed"])
        self.mortar_g = float(weapons["mortar"]["gravity"])
        self.mortar_splash = float(weapons["mortar"]["splash_radius"])
        self.light_range = float(weapons["lightning"]["range"])
        self.drop_r = float(cfg["drops"]["radius"])
        self.seed = int(hello.get("seed", 1))
        self.slot = int(hello.get("slot", 0))
        self.stuck = 0
        self.bias = 0.0
        self.hurt_until = -1.0
        self.hurt_bearing = None
        self.scan = (self.seed % 628) / 100.0
        mf.log("oval-ghost slot", self.slot)

    def act(self, obs):
        me = obs["self"]
        x, y, z = (float(v) for v in me["pos"])
        stats = me.get("stats") or {}
        walk = float(stats.get("walk_speed", self.walk))
        energy = float(me.get("energy", 0))
        t = float(obs.get("t", 0))
        zone = obs.get("zone") or {}
        my_id = me.get("id")

        for ev in obs.get("events") or []:
            if ev.get("type") != "damaged":
                continue
            if ev.get("weapon") == "zone" or ev.get("bearing") is None:
                continue
            self.hurt_until = t + 3.0
            self.hurt_bearing = float(ev["bearing"])

        enemies = []
        npcs = []
        for actor in obs.get("mechs") or []:
            if actor.get("kind") == "npc":
                npcs.append(actor)
            elif actor.get("kind") == "mech":
                enemies.append(actor)

        def dist_of(pos):
            return _hypot(float(pos[0]) - x, float(pos[1]) - y)

        enemies.sort(key=lambda m: dist_of(m["pos"]))
        closest = dist_of(enemies[0]["pos"]) if enemies else 1e9
        n_near = sum(1 for m in enemies if dist_of(m["pos"]) < 80.0)
        npc_close = any(dist_of(m["pos"]) < self.npc_aggro for m in npcs)

        weapons = list(me.get("weapons") or [])
        while len(weapons) < 2:
            weapons.append(None)
        held = {w.get("kind") for w in weapons if w}
        empty_slot = any(w is None for w in weapons[:2])

        gx, gy, need = self._goal(x, y, zone, enemies)
        in_zone = bool(me.get("in_zone", True))
        cx, cy = _pair(zone.get("center"), (x, y))
        radius = float(zone.get("radius", 1e9))
        edge = radius - _hypot(x - cx, y - cy)
        dps = float(zone.get("dps", 0))
        se = float(zone.get("shrink_end", t + 999))
        if not in_zone:
            time_left = 0.0
        elif t < se:
            time_left = se - t
        else:
            time_left = 999.0

        if not in_zone:
            mode = "boost"
        elif need <= 5.0:
            mode = "hold"
        elif time_left > need / max(walk, 0.1) + 12.0:
            mode = "hold"
        elif time_left > need / max(walk, 0.1) + 1.5:
            mode = "walk"
        else:
            mode = "boost"
        if in_zone and need > 5.0 and edge < 18.0 and mode == "hold":
            mode = "walk"
        if in_zone and edge < 7.0 and need > 2.0:
            mode = "boost"

        threatened = self._threatened(x, y, t, enemies)
        fx, fy = self._flee(x, y, enemies, npcs, obs.get("sounds") or [])
        dox, doy = self._dodge(x, y, z, obs.get("projectiles") or [])

        zx, zy = gx - x, gy - y
        zd = _hypot(zx, zy)
        if zd > 0.5:
            zx, zy = zx / zd, zy / zd
        else:
            zx = zy = 0.0

        if not in_zone:
            zw = 6.0
        elif mode == "boost":
            zw = 3.2
        elif mode == "walk":
            zw = 1.5
        else:
            zw = 0.0

        if closest < 28:
            fw = 2.4
        elif closest < 50:
            fw = 1.6
        elif closest < 100:
            fw = 1.1
        elif closest < 160 and mode == "hold":
            fw = 0.45
        else:
            fw = 0.0
        if threatened:
            fw = max(fw, 1.8)
            if mode == "hold":
                mode = "walk"
        if npc_close:
            fw = max(fw, 0.9)
        if dps >= 80 or edge < 12:
            fw *= 0.4
        if not in_zone:
            fw *= 0.15

        dw = 2.0
        mx = zw * zx + fw * fx + dw * dox
        my = zw * zy + fw * fy + dw * doy

        take_id = None
        if fw < 1.2 and mode != "boost" and in_zone:
            take_id = self._take_id(x, y, z, obs.get("pickups") or [], held, empty_slot)
            if mode == "hold" and closest > 75.0 and not npc_close:
                mx, my = self._loot_nudge(
                    x, y, z, mx, my, obs, me, zone, held, empty_slot, enemies, closest
                )

        vx, vy = float(me["vel"][0]), float(me["vel"][1])
        spd = _hypot(vx, vy)
        if _hypot(mx, my) > 0.45 and spd < 1.2 and (need > 8.0 or not in_zone):
            self.stuck += 1
        else:
            self.stuck = max(0, self.stuck - 1)
        jump = False
        if self.stuck > 7:
            self.bias = (self.bias + 1.15) if self.bias < 2.4 else -1.6
            jump = bool(me.get("grounded", True))
        if self.bias:
            c, s = math.cos(self.bias), math.sin(self.bias)
            mx, my = mx * c - my * s, mx * s + my * c
            if self.stuck == 0:
                self.bias *= 0.4
                if abs(self.bias) < 0.12:
                    self.bias = 0.0

        move, boost = self._throttle(
            mx, my, mode, energy, in_zone, dps, closest, walk, _hypot(dox, doy), fw, threatened
        )

        fight = self._should_fight(closest, n_near, t, float(zone.get("radius", 9999)))
        aim, fire, mortar_aim = self._combat(
            x, y, z, me, enemies, weapons, fight, closest, energy
        )
        if mortar_aim is not None and not any(
            w and w.get("kind") in ("gun", "lightning") and fire[i]
            for i, w in enumerate(weapons[:2])
        ):
            aim = mortar_aim

        reload = False
        if not fight and closest > 55.0:
            for w in weapons[:2]:
                if not w or w.get("kind") != "gun":
                    continue
                ammo = float(w.get("ammo", 0))
                mag = float((w.get("stats") or {}).get("mag", 12))
                if float(w.get("reload", 0)) <= 0 and ammo <= max(2.0, mag * 0.25):
                    reload = True

        act = {
            "move": [round(move[0], 4), round(move[1], 4)],
            "boost": boost,
            "aim": [round(aim[0], 4), round(aim[1], 4)],
            "fire": fire,
        }
        if jump:
            act["jump"] = True
        if reload:
            act["reload"] = True
        if take_id is not None:
            act["take"] = take_id
        offer = obs.get("offer")
        if offer and offer.get("choices"):
            act["pick"] = self._pick(offer["choices"])
        return act

    def _goal(self, x, y, zone, enemies):
        nx, ny = _pair(zone.get("next_center"), _pair(zone.get("center"), (x, y)))
        rn = float(zone.get("next_radius", 0))
        dist_next = _hypot(x - nx, y - ny)
        if rn < 12.0:
            return nx, ny, dist_next
        margin = max(6.0, min(25.0, rn * 0.08))
        safe = max(0.0, rn - margin)
        if dist_next <= safe:
            return x, y, 0.0
        k = safe / dist_next
        gx = nx + (x - nx) * k
        gy = ny + (y - ny) * k
        if enemies:
            e = min(enemies, key=lambda m: _hypot(m["pos"][0] - x, m["pos"][1] - y))
            # Sidestep the nearest mech while still entering along the radius.
            inward_x, inward_y = nx - x, ny - y
            il = _hypot(inward_x, inward_y) or 1.0
            px, py = -inward_y / il, inward_x / il
            ex, ey = float(e["pos"][0]), float(e["pos"][1])
            if _hypot(x + px * 20 - ex, y + py * 20 - ey) < _hypot(x - px * 20 - ex, y - py * 20 - ey):
                px, py = -px, -py
            gx += px * min(30.0, safe * 0.15)
            gy += py * min(30.0, safe * 0.15)
            gx, gy = _clamp_disk(gx, gy, nx, ny, safe)
        return gx, gy, _hypot(gx - x, gy - y)

    def _threatened(self, x, y, t, enemies):
        """True when someone is shooting us or lining a beam up on us."""
        if t < self.hurt_until:
            return True
        for m in enemies:
            mx, my = float(m["pos"][0]), float(m["pos"][1])
            dist = _hypot(mx - x, my - y)
            if dist > 240.0:
                continue
            aim = m.get("aim") or [0.0, 0.0]
            to_us = math.atan2(y - my, x - mx)
            off = abs(_angdiff(float(aim[0]), to_us))
            if m.get("charging") and off < 0.45:
                return True
            if off < 0.2 and dist < 180.0:
                return True
        return False

    def _flee(self, x, y, enemies, npcs, sounds):
        fx = fy = 0.0
        for m in enemies:
            mx, my = float(m["pos"][0]), float(m["pos"][1])
            dx, dy = x - mx, y - my
            dist = _hypot(dx, dy) or 1.0
            if dist > 230.0:
                continue
            w = max(0.2, (230.0 - dist) / 230.0)
            aim = m.get("aim") or [0.0, 0.0]
            to_us = math.atan2(y - my, x - mx)
            off = abs(_angdiff(float(aim[0]), to_us))
            aiming = off < 0.35 or bool(m.get("charging"))
            if aiming:
                # Leave the beam: strafe harder than we back straight up.
                sx, sy = -dy / dist, dx / dist
                if (int(m.get("id", 0)) + self.slot) % 2 == 0:
                    sx, sy = -sx, -sy
                w *= 1.8
                fx += sx * w * 1.4 + (dx / dist) * w * 0.5
                fy += sy * w * 1.4 + (dy / dist) * w * 0.5
            else:
                fx += dx / dist * w
                fy += dy / dist * w
        for m in npcs:
            mx, my = float(m["pos"][0]), float(m["pos"][1])
            dx, dy = x - mx, y - my
            dist = _hypot(dx, dy) or 1.0
            reach = self.npc_aggro + 10.0
            if dist < reach:
                w = (reach - dist) / reach
                fx += dx / dist * w
                fy += dy / dist * w
        for s in sounds:
            if s.get("dist") != "near":
                continue
            kind = s.get("kind")
            if kind not in (
                "step", "boost", "gun", "laser", "laser_charge",
                "lightning", "mortar", "jet", "explosion",
            ):
                continue
            bearing = float(s.get("bearing", 0.0))
            w = 0.85 if kind in ("gun", "laser", "lightning", "mortar", "explosion") else 0.4
            fx -= math.cos(bearing) * w
            fy -= math.sin(bearing) * w
        mag = _hypot(fx, fy)
        if mag > 1e-6:
            return fx / mag, fy / mag
        return 0.0, 0.0

    def _dodge(self, x, y, z, projectiles):
        dox = doy = 0.0
        mz = z + 4.5
        for p in projectiles:
            pos = p.get("pos") or [0, 0, 0]
            vel = p.get("vel") or [0, 0, 0]
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
            rx, ry, rz = x - px, y - py, mz - pz
            v2 = vx * vx + vy * vy + vz * vz
            if v2 < 1.0:
                continue
            tca = -(rx * vx + ry * vy + rz * vz) / v2
            if tca < -0.05 or tca > 1.4:
                continue
            cx = rx + vx * tca
            cy = ry + vy * tca
            cz = rz + vz * tca
            miss = math.sqrt(cx * cx + cy * cy + cz * cz)
            thresh = 14.0 if p.get("kind") == "mortar" else 8.0
            if miss > thresh:
                continue
            plen = _hypot(vx, vy) or 1.0
            sx, sy = -vy / plen, vx / plen
            if rx * sx + ry * sy < 0:
                sx, sy = -sx, -sy
            strength = (thresh - miss) / thresh * max(0.25, 1.4 - max(tca, 0.0))
            dox += sx * strength
            doy += sy * strength
        mag = _hypot(dox, doy)
        if mag > 1e-6:
            # Keep a little magnitude so a near-miss dodges harder than a graze.
            scale = min(1.0, mag)
            return dox / mag * scale, doy / mag * scale
        return 0.0, 0.0

    def _take_id(self, x, y, z, pickups, held, empty_slot):
        if not empty_slot:
            return None
        best = None
        best_s = 99
        for p in pickups:
            pos = p.get("pos") or [0, 0, 0]
            kind = p.get("kind")
            if kind in held:
                continue
            if abs(float(pos[2]) - z) > 4.0:
                continue
            if _hypot(float(pos[0]) - x, float(pos[1]) - y) > self.pickup_r + 0.25:
                continue
            score = LOOT_RANK.get(kind, 5)
            if score < best_s:
                best_s = score
                best = p.get("id")
        return best

    def _loot_nudge(self, x, y, z, mx, my, obs, me, zone, held, empty_slot, enemies, closest):
        nx, ny = _pair(zone.get("next_center"), (x, y))
        rn = float(zone.get("next_radius", 0))
        target = None
        td = 1e9
        for c in obs.get("cubes") or []:
            pos = c.get("pos") or [0, 0, 0]
            dist = _hypot(float(pos[0]) - x, float(pos[1]) - y)
            if dist < 16.0 and dist < td and abs(float(pos[2]) - z) < 6.0:
                td = dist
                target = pos
        if empty_slot:
            for p in obs.get("pickups") or []:
                pos = p.get("pos") or [0, 0, 0]
                if p.get("kind") in held:
                    continue
                dist = _hypot(float(pos[0]) - x, float(pos[1]) - y)
                if dist < 14.0 and dist < td and abs(float(pos[2]) - z) < 6.0:
                    td = dist
                    target = pos
        for d in obs.get("drops") or []:
            if float(d.get("lands_in", 1)) > 0.3:
                continue
            holder = d.get("holder")
            if holder not in (None, me.get("id")):
                continue
            if enemies:
                continue
            pos = d.get("pos") or [0, 0, 0]
            if rn > 12 and _hypot(float(pos[0]) - nx, float(pos[1]) - ny) > rn - 6:
                continue
            dist = _hypot(float(pos[0]) - x, float(pos[1]) - y)
            ring = float(d.get("radius", self.drop_r))
            if dist < 36.0:
                if dist <= ring:
                    return 0.0, 0.0
                target = pos
                td = dist
        if target is None or td < 1.2:
            return mx, my
        return (float(target[0]) - x) / td, (float(target[1]) - y) / td

    def _throttle(self, mx, my, mode, energy, in_zone, dps, closest, walk, dodge, fw, threatened):
        md = _hypot(mx, my)
        if md < 0.05:
            return [0.0, 0.0], False
        ux, uy = mx / md, my / md
        urgent = mode == "boost" or closest < 36.0 or dodge > 0.55 or threatened
        if urgent:
            need_boost = (
                mode == "boost"
                or not in_zone
                or closest < 45.0
                or (threatened and closest < 190.0 and energy > 18.0)
                or dps >= 50.0
            )
            boost = bool(need_boost and (energy > 8.0 or not in_zone))
            return [ux, uy], boost
        if mode == "walk" or fw >= 0.7 or dodge > 0.45:
            return [ux, uy], False
        cap = max(0.12, (self.step_speed - 0.5) / max(walk, 0.1))
        return [ux * min(1.0, cap), uy * min(1.0, cap)], False

    def _should_fight(self, closest, n_near, t, radius):
        if closest >= 1e8:
            return False
        if closest < 34.0 and n_near <= 1:
            return True
        if radius < 85.0 and closest < 160.0 and n_near <= 2:
            return True
        if t < self.hurt_until and closest < 70.0 and n_near <= 1:
            return True
        return False

    def _combat(self, x, y, z, me, enemies, weapons, fight, closest, energy):
        if enemies:
            aim = self._aim_gun(x, y, z, me, enemies[0])
        elif self.hurt_bearing is not None:
            aim = [self.hurt_bearing, 0.0]
        else:
            self.scan = _wrap(self.scan + 0.45)
            aim = [self.scan, 0.02]
        fire = [False, False]
        mortar_aim = None
        if not fight or not enemies:
            # Releasing a charged laser fires it. Keep holding only if already charged
            # a meaningful amount, then let the shot go on purpose along our aim.
            for i, w in enumerate(weapons[:2]):
                if w and w.get("kind") == "laser" and float(w.get("charge") or 0) >= 0.2:
                    charge_need = float((w.get("stats") or {}).get("charge", 0.75))
                    fire[i] = float(w.get("charge") or 0) < charge_need * 0.9
            return aim, fire, None
        enemy = enemies[0]
        for i, w in enumerate(weapons[:2]):
            if not w:
                continue
            kind = w.get("kind")
            wst = w.get("stats") or {}
            if kind == "gun":
                rng = float(wst.get("range", self.gun_range))
                if float(w.get("ammo", 0)) > 0 and float(w.get("reload", 0)) <= 0 and closest < rng:
                    fire[i] = True
            elif kind == "lightning":
                rng = float(wst.get("range", self.light_range))
                if closest < rng and energy > 12.0:
                    fire[i] = True
            elif kind == "mortar":
                splash = float(wst.get("splash_radius", self.mortar_splash))
                if closest > splash * 3.5 and closest < 210.0 and float(w.get("cooldown", 1)) <= 0.05:
                    fire[i] = True
                    mortar_aim = self._aim_mortar(x, y, z, me, enemy, wst)
            elif kind == "laser":
                if closest > 18.0:
                    charge_need = float(wst.get("charge", 0.75))
                    held = float(w.get("charge") or 0)
                    fire[i] = held < charge_need * 0.92
        return aim, fire, mortar_aim

    def _aim_gun(self, x, y, z, me, enemy):
        ex, ey, ez = (float(v) for v in enemy["pos"])
        ev = enemy.get("vel") or [0.0, 0.0, 0.0]
        muzzle_z = z + float((me.get("stats") or {}).get("muzzle_height", 7.0))
        tx, ty, tz = ex, ey, ez + 4.5
        speed = max(self.gun_speed, 1.0)
        for _ in range(2):
            flight = _hypot(tx - x, ty - y) / speed
            tx = ex + float(ev[0]) * flight
            ty = ey + float(ev[1]) * flight
            tz = ez + 4.5 + float(ev[2]) * flight
        return self._yaw_pitch(x, y, muzzle_z, tx, ty, tz)

    def _aim_mortar(self, x, y, z, me, enemy, wst):
        ex, ey, ez = (float(v) for v in enemy["pos"])
        ev = enemy.get("vel") or [0.0, 0.0, 0.0]
        speed = float(wst.get("speed", self.mortar_speed))
        gravity = float(wst.get("gravity", self.mortar_g))
        muzzle_z = z + float((me.get("stats") or {}).get("muzzle_height", 7.0)) + 2.5
        # One lead guess using a flat-fire time, then loft for that ground range.
        dist = _hypot(ex - x, ey - y) or 1.0
        flight = dist / max(speed, 1.0) * 1.4
        tx = ex + float(ev[0]) * flight
        ty = ey + float(ev[1]) * flight
        dist = _hypot(tx - x, ty - y)
        yaw = math.atan2(ty - y, tx - x)
        sin2 = dist * gravity / max(speed * speed, 1.0)
        if sin2 >= 1.0:
            pitch = math.pi / 4
        else:
            pitch = 0.5 * math.asin(max(0.0, sin2))
        # Aim a touch higher when the target is above the muzzle.
        pitch += math.atan2((ez + 1.0) - muzzle_z, dist) * 0.25
        pitch = max(self.min_pitch, min(self.max_pitch, pitch))
        return [yaw, pitch]

    def _yaw_pitch(self, x, y, muzzle_z, tx, ty, tz):
        dx, dy = tx - x, ty - y
        horiz = _hypot(dx, dy) or 0.001
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(tz - muzzle_z, horiz)
        pitch = max(self.min_pitch, min(self.max_pitch, pitch))
        return [yaw, pitch]

    def _pick(self, choices):
        best_i, best_r = 0, 10 ** 9
        for i, choice in enumerate(choices):
            rank = UPGRADE_RANK.get(choice.get("id"), 40)
            if rank < best_r:
                best_r = rank
                best_i = i
        return best_i


def _pair(value, default):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return default


mf.run(Bot)
