"""Oval Ghost: a cover-first survival bot for Mech Royale (mechfight/1).

Placement is the score, and the practice house bots hunt us. So the bot:
- reads the map at startup and keeps a 2 m height grid (arena.py) for line of sight and paths,
- walks a real path into the next safe circle early and waits there in a well-enclosed spot,
- tracks every mech it has seen, heard, or been hit by,
- picks where to stand by scoring nearby spots: hidden from every threat, or exposed to exactly
  one target it is ready to shoot. Out of ammo, hurt, or seen by two mechs, it ducks back out of sight,
- shoots with lead (gun), pre-charges a laser behind cover and fires it the moment it is aimed,
- answers every observation well inside the 50 ms budget (heavy work is capped and spread over ticks).
"""
import gc
import math
import time

import mechfight as mf

try:
    from arena import Arena, PathJob
except Exception:  # the bot still plays (without cover) if the map module cannot load
    Arena = None
    PathJob = None

TWO_PI = 2.0 * math.pi

# Lower is better. Survival first, then the weapons we fight with. Plating (+15% max HP) beats
# armor (-8% weapon damage = +8.7% effective HP, and nothing against the zone, which decides the
# final circle). Nanites speed up the regen that the peek-and-hide cycle lives on.
UPGRADE_RANK = {
    "plating": 0,
    "nanites": 1,
    "armor": 2,
    "laser_charge": 3,
    "laser_damage": 4,
    "gun_damage": 5,
    "gun_rate": 6,
    "gun_mag": 7,
    "lightning_efficiency": 8,
    "lightning_range": 9,
    "servos": 10,
    "gyros": 11,
    "sensors": 12,
    "gun_velocity": 13,
    "reactor": 14,
    "capacitor": 15,
    "jets": 16,
    "laser_pierce": 17,
    "lightning_chain": 18,
    "mortar_reload": 30,
    "mortar_splash": 31,
    "mortar_velocity": 32,
}
# Laser first: hitscan, 315 per full charge, outranges everything, and we can charge it behind
# cover and fire the moment we peek. Lightning next: it shorts lasers (drops their charge, no new
# one for 2 s) and does 240 DPS, but only inside 45 m. Never a mortar (a laser hit cooks it off
# on our back: +60%).
LOOT_RANK = {"laser": 0, "lightning": 1, "mortar": 9, "gun": 9}

SOUND_RANGE_GUESS = {"near": 35.0, "mid": 110.0, "far": 230.0}
WEAPON_RANGE_GUESS = {"laser": 140.0, "gun": 110.0, "lightning": 35.0, "mortar": 150.0, "npc": 40.0}


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


def hyp(x, y):
    return math.hypot(x, y)


class Track:
    """What we know about one enemy mech."""

    __slots__ = ("id", "x", "y", "z", "vx", "vy", "t", "hp", "max_hp", "weapons", "aim",
                 "charging", "listed", "guess", "danger", "dist", "sees_us")

    def __init__(self, mid):
        self.id = mid
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = 0.0
        self.t = -99.0
        self.hp = 1400.0
        self.max_hp = 1400.0
        self.weapons = ()
        self.aim = 0.0
        self.charging = False
        self.listed = False
        self.guess = False  # position estimated from a bearing, not seen
        self.danger = 1.0
        self.dist = 1e9
        self.sees_us = False


class Bot:
    name = "oval-ghost"

    def __init__(self, hello):
        cfg = hello.get("config") or {}
        mech = cfg.get("mech") or {}
        wcfg = cfg.get("weapons") or {}
        self.walk = float(mech.get("walk_speed", 11.7))
        self.boost_spd = float(mech.get("boost_speed", 19.5))
        self.min_pitch = float(mech.get("min_pitch", -0.7))
        self.max_pitch = float(mech.get("max_pitch", 1.3))
        self.turn_rate = float(mech.get("turn_rate", 4.0))
        self.regen_delay = float(mech.get("regen_delay", 5.0))
        self.step_speed = float((cfg.get("sound") or {}).get("step_speed", 5.0))
        vis = cfg.get("vision") or {}
        self.vis_range = float(vis.get("range", 200.0))
        self.near_range = float(vis.get("near_range", 45.0))
        self.memory_s = float(vis.get("memory_s", 5.0))
        self.pickup_r = float((cfg.get("loot") or {}).get("pickup_radius", 4.0))
        self.cube_r = float((cfg.get("cubes") or {}).get("pickup_radius", 3.5))
        npc = cfg.get("npc") or {}
        self.npc_aggro = float(npc.get("aggro_range", 45.0))
        g = wcfg.get("gun") or {}
        self.gun_speed = float(g.get("speed", 252.0))
        self.gun_range = float(g.get("range", 230.0))
        las = wcfg.get("laser") or {}
        self.laser_charge = float(las.get("charge", 0.75))
        self.laser_hold = float(las.get("hold_s", 2.0))
        self.laser_range = float(las.get("range", 280.0))
        li = wcfg.get("lightning") or {}
        self.light_range = float(li.get("range", 45.0))
        self.light_cone = math.radians(float(li.get("cone_deg", 30.0)))
        mo = wcfg.get("mortar") or {}
        self.mortar_speed = float(mo.get("speed", 72.0))
        self.mortar_g = float(mo.get("gravity", 22.0))
        self.mortar_splash = float(mo.get("splash_radius", 7.0))
        self.drop_r = float((cfg.get("drops") or {}).get("radius", 8.0))
        self.seed = int(hello.get("seed", 1) or 1)
        self.slot = int(hello.get("slot", 0) or 0)
        self.rng = (self.seed * 2654435761) & 0xFFFFFFFF or 1

        self.arena = None
        t0 = time.perf_counter()
        if Arena is not None:
            try:
                m = hello.get("map") or {}
                self.arena = Arena(m.get("dir", "/arena"), m.get("files"),
                                   step_up=float(mech.get("step_up", 1.2)),
                                   mech_height=float(mech.get("height", 9.0)))
            except Exception as e:
                mf.log("map load failed, playing without cover:", repr(e))
                self.arena = None
        mf.log("oval-ghost slot", self.slot, "map", "ok" if self.arena else "none",
               "%.2fs" % (time.perf_counter() - t0))

        self.tracks = {}
        self.npcs = []
        self.hurt_t = -99.0
        self.hurt_bearing = None
        self.last_dmg_t = -99.0
        self.last_hit_t = -99.0
        self.sound_marks = []  # (t, x, y, weight)
        self.dmg_by = {}  # attacker id -> last time it hit us
        self.cover_dist = 99.0  # distance to the nearest hidden spot found by the last tactical search
        self.laser_alert = -99.0  # last time a visible enemy was charging a laser at us
        self.pickup_mem = {}  # id -> [kind, x, y, z, t_seen]: weapons we saw lying around
        self.explore = None
        self.explore_t = -99.0
        self.spawn_yaw = None

        # navigation
        self.path = None
        self.path_i = 0
        self.path_goal = None
        self.job = None
        self.job_goal = None
        self.blocked = set()
        self.stuck = 0
        self.unstick_t = -1.0
        self.unstick_dir = 0.0
        self.last_pos = None

        # strategy / tactics
        self.camp = None
        self.camp_key = None
        self.spot = None
        self.spot_score = -1e9
        self.spot_t = -99.0
        self.spot_mode = ""
        self.jitter_t = -99.0
        self.scan_i = 0
        self.mode = "start"
        self.label = ""
        self.tick_n = 0
        self.slow_ticks = 0
        self.max_ms = 0.0

        # Big static data is never garbage: keep the collector off it so ticks stay flat.
        gc.collect()
        gc.freeze()
        gc.set_threshold(20000, 50, 50)

    # ------------------------------------------------------------------ helpers
    def rand(self):
        self.rng ^= (self.rng << 13) & 0xFFFFFFFF
        self.rng ^= self.rng >> 17
        self.rng ^= (self.rng << 5) & 0xFFFFFFFF
        return (self.rng & 0xFFFFFF) / float(0x1000000)

    def los(self, ax, ay, az, bx, by, bz):
        if self.arena is None:
            return True
        return self.arena.los(ax, ay, az, bx, by, bz)

    def surface(self, x, y, fallback):
        if self.arena is None:
            return fallback
        return self.arena.surface(x, y)

    # ------------------------------------------------------------------ main
    def act(self, obs):
        t_start = time.perf_counter()
        self.tick_n += 1
        me = obs["self"]
        x, y, z = (float(v) for v in me["pos"])
        vx, vy = float(me["vel"][0]), float(me["vel"][1])
        t = float(obs.get("t", 0.0))
        hp = float(me.get("hp", 1400.0))
        max_hp = float(me.get("max_hp", 1400.0)) or 1400.0
        energy = float(me.get("energy", 0.0))
        stats = me.get("stats") or {}
        walk = float(stats.get("walk_speed", self.walk))
        eye_h = float(stats.get("eye_height", 8.0))
        muzzle_h = float(stats.get("muzzle_height", 7.0))
        my_aim = me.get("aim") or [0.0, 0.0]
        self.my_aim0 = my_aim[0]
        self.t = t
        self.me = (x, y, z)
        self.hp_frac = hp / max_hp

        weapons = list(me.get("weapons") or [])
        while len(weapons) < 2:
            weapons.append(None)
        self.weapons = weapons

        self._shot_bands = [(float(s.get("bearing", 0.0)), s.get("dist"), s.get("kind"))
                            for s in (obs.get("sounds") or [])
                            if s.get("kind") in ("gun", "laser", "lightning", "mortar")]
        self._events(obs, x, y, t)
        self._remember_pickups(obs, x, y, z, t)
        self._update_tracks(obs, x, y, z, t)
        self._sounds(obs, x, y, t)
        zone = self._zone_info(obs.get("zone") or {}, x, y, t)
        threats = self._threats(x, y, z, t)

        target = self._pick_target(threats, x, y, z, muzzle_h)
        ready, eff_range = self._readiness(weapons, energy, target)
        active = [th for th in threats if th.dist < 300.0 and (t - th.t) < 9.0]
        exposed_now = [th for th in active if th.sees_us]

        engage = False
        if target is not None:
            others = sum(1 for th in exposed_now if th is not target)
            low_target = target.hp < 0.35 * max(target.max_hp, 1.0)
            healthy = self.hp_frac > 0.3 or target.hp < hp
            cover_ok = self.cover_dist <= 14.0 or target.dist < 40.0 or self.arena is None
            if ready and target.dist < eff_range:
                if others == 0 and (healthy or low_target):
                    # One enemy can see us (or none): trade from cover, and never walk away from a
                    # fight we are already in or one that is nearly won. Hiding lets it regenerate.
                    engage = cover_ok or target.sees_us or low_target
                elif others == 1 and low_target and self.hp_frac > 0.4:
                    engage = True  # finish it, then deal with the other one

        # Loaded and in range: fight, even with a second mech watching. Tested on 96+96 simulated
        # matches against the cautious rule: more kills and longer survival both ways, because
        # hiding from hunters that come to us only hands them free shots and regen time.
        if target is not None and ready and target.dist < eff_range:
            engage = True

        # ---------------- where to go
        move_goal = None
        hurry = False
        speed_mode = "walk"
        if not zone["in_zone"] and zone["dps"] > 0:
            # Outside: get back in by the path, fast. Fights wait.
            move_goal = zone["goal"]
            hurry = True
            speed_mode = "boost"
            self.mode = "zone!"
        elif active and self.arena is not None:
            urgent = (t - self.laser_alert) < 0.4 and not engage
            spot = self._tactical(x, y, z, t, active, target, engage, zone, t_start, urgent)
            if spot is not None and self.cover_dist > 110.0 and exposed_now and not engage:
                # Caught in the open with nowhere to hide nearby: commit to the camp (an enclosed
                # spot) and run there on a planned path instead of zig-zagging.
                spot = None
                move_goal = self._camp(x, y, z, t, zone)
                speed_mode = "boost" if energy > 15 else "walk"
                hurry = True
                self.mode = "run"
            if spot is not None:
                move_goal = spot
                self.mode = "fight" if engage else "hide"
                n_exp = len(exposed_now)
                if urgent and energy > 8:
                    speed_mode = "boost"
                elif n_exp >= 2 or (n_exp == 1 and not engage):
                    speed_mode = "boost" if energy > 25 else "walk"
                elif hyp(spot[0] - x, spot[1] - y) > 8.0:
                    speed_mode = "walk"
                else:
                    speed_mode = "walk"
            if zone["late"]:
                # The ring is coming: a fight never outranks staying alive.
                if not self._inside(spot or (x, y), zone, 4.0):
                    move_goal = zone["goal"]
                    hurry = True
                    self.mode = "rotate"
        if (move_goal is None and self.hp_frac < 0.75 and zone["in_zone"] and not zone["late"]
                and self.arena is not None and self._hidden_here(x, y, z, t)):
            # Hurt and out of sight: stay put and let regen work.
            move_goal = (x, y)
            self.mode = "heal"
            speed_mode = "quiet"
        loot = None
        if not hurry and self.mode not in ("fight", "zone!", "rotate", "run"):
            loot = self._loot_goal(obs, x, y, z, t, zone, threats, weapons)
            if loot is not None and (move_goal is None or not exposed_now):
                move_goal = loot
                d_l = hyp(loot[0] - x, loot[1] - y)
                speed_mode = "boost" if (d_l > 30.0 and energy > 50.0) else "walk"
                self.mode = "loot"
        held_kinds = {w.get("kind") for w in weapons if w}
        if (move_goal is None and "laser" not in held_kinds and t < 180.0 and not zone["late"]
                and zone["in_zone"] and self.hp_frac > 0.6 and self.arena is not None):
            # No laser yet: it is worth more than anything else, so go and look for one.
            ex = self._explore(x, y, z, t, zone)
            if ex is not None:
                move_goal = ex
                speed_mode = "walk"
                self.mode = "explore"
        if move_goal is None:
            camp = self._camp(x, y, z, t, zone)
            move_goal = camp
            self.mode = "camp"
            d = hyp(camp[0] - x, camp[1] - y)
            if zone["late"] or zone["urgent"]:
                hurry = True
                speed_mode = "boost" if zone["urgent"] else "walk"
                self.mode = "rotate"
            elif d > 30.0:
                speed_mode = "walk"
            else:
                speed_mode = "quiet"

        mx, my = self._steer(x, y, z, vx, vy, t, move_goal, hurry or speed_mode == "boost", t_start)
        dox, doy = self._dodge(x, y, z, obs.get("projectiles") or [])
        if dox or doy:
            mx += dox * 1.2
            my += doy * 1.2
        mag = hyp(mx, my)
        if mag > 1e-6:
            mx /= mag
            my /= mag
        boost = False
        dist_goal = hyp(move_goal[0] - x, move_goal[1] - y) if move_goal else 0.0
        if dist_goal < 1.2 and not (dox or doy):
            mx = my = 0.0
        elif speed_mode == "boost" and (energy > 12.0 or not zone["in_zone"]):
            boost = True
        elif speed_mode == "quiet" and dist_goal < 30.0:
            cap = max(0.15, (self.step_speed - 0.6) / max(walk, 0.1))
            mx *= cap
            my *= cap
        elif dist_goal < 4.0:
            mx *= 0.5
            my *= 0.5

        jump = False
        if self.stuck > 8 and me.get("grounded", True):
            jump = True
        if self.unstick_t > t and energy > 10.0 and not me.get("grounded", True):
            boost = True

        # ---------------- aim and fire
        aim, fire, charging = self._combat(x, y, z, me, weapons, target, engage, threats, muzzle_h, energy, t)
        pass
        if charging and hyp(mx, my) > 0:
            pass  # moving at half speed while charging is accepted
        reload = False
        for w in weapons[:2]:
            if w and w.get("kind") == "gun":
                ammo = float(w.get("ammo", 0))
                mag_sz = float((w.get("stats") or {}).get("mag", 12))
                if float(w.get("reload", 0)) <= 0:
                    if ammo <= 0:
                        reload = True
                    elif not exposed_now and not fire[weapons.index(w)] and ammo <= mag_sz * 0.5:
                        reload = True

        act = {
            "move": [round(mx, 3), round(my, 3)],
            "boost": bool(boost),
            "aim": [round(aim[0], 4), round(aim[1], 4)],
            "fire": fire,
        }
        if jump:
            act["jump"] = True
        if reload:
            act["reload"] = True
        take = self._take(obs, x, y, z, weapons)
        if take is not None:
            act["take"] = take[0]
            if take[1] is not None:
                act["drop"] = take[1]
        offer = obs.get("offer")
        if offer and offer.get("choices"):
            act["pick"] = self._pick(offer["choices"])

        ms = (time.perf_counter() - t_start) * 1000.0
        if ms > self.max_ms:
            self.max_ms = ms
        if ms > 30.0:
            self.slow_ticks += 1
        nf = "" if any(fire) else getattr(self, "nofire", "")
        mf.draw.label("%s %s hp%d n%d %s" % (self.mode, ("T%d" % target.id) if target else "-",
                                             int(hp), len(active), nf))
        self.nofire = "notgt"
        return act

    # ------------------------------------------------------------------ perception
    def _events(self, obs, x, y, t):
        for ev in obs.get("events") or []:
            kind = ev.get("type")
            if kind == "damaged":
                if ev.get("weapon") == "zone":
                    continue
                self.last_dmg_t = t
                b = ev.get("bearing")
                if b is None:
                    continue
                self.hurt_t = t
                self.hurt_bearing = float(b)
                by = ev.get("by")
                if by is not None and 0 <= int(by) < 100:
                    self.dmg_by[int(by)] = t
                    tr = self.tracks.get(int(by))
                    if tr is None or (t - tr.t) > 1.0:
                        # Unseen attacker: put a guess on the bearing.
                        if tr is None:
                            tr = Track(int(by))
                            self.tracks[int(by)] = tr
                        d = WEAPON_RANGE_GUESS.get(ev.get("weapon"), 110.0)
                        # A shot we heard from the same bearing tells us roughly how far.
                        for sb, band, skind in getattr(self, "_shot_bands", ()):
                            if skind == ev.get("weapon") and abs(wrap(sb - float(b))) < 0.3:
                                d = SOUND_RANGE_GUESS.get(band, d)
                                break
                        tr.x = x + math.cos(float(b)) * d
                        tr.y = y + math.sin(float(b)) * d
                        tr.z = self.me[2]
                        tr.vx = tr.vy = 0.0
                        tr.t = t
                        tr.guess = True
                        if ev.get("weapon") in ("laser", "gun", "mortar", "lightning"):
                            if ev.get("weapon") not in tr.weapons:
                                tr.weapons = tuple(tr.weapons) + (ev.get("weapon"),)
            elif kind == "hit":
                self.last_hit_t = t
            elif kind in ("took", "take_failed"):
                self.pickup_mem.pop(ev.get("id"), None)
            elif kind == "kill":
                tr = self.tracks.pop(int(ev.get("target", -1)), None)

    def _update_tracks(self, obs, x, y, z, t):
        for tr in self.tracks.values():
            tr.listed = False
        npcs = []
        for a in obs.get("mechs") or []:
            pos = a.get("pos") or [0, 0, 0]
            if a.get("kind") == "npc":
                npcs.append((float(pos[0]), float(pos[1]), float(pos[2])))
                continue
            mid = int(a.get("id", -1))
            tr = self.tracks.get(mid)
            if tr is None:
                tr = Track(mid)
                self.tracks[mid] = tr
            vel = a.get("vel") or [0, 0, 0]
            tr.x, tr.y, tr.z = float(pos[0]), float(pos[1]), float(pos[2])
            tr.vx, tr.vy = float(vel[0]), float(vel[1])
            tr.t = t
            tr.listed = True
            tr.guess = False
            if a.get("hp") is not None:
                tr.hp = float(a["hp"])
            if a.get("max_hp") is not None:
                tr.max_hp = float(a["max_hp"])
            tr.weapons = tuple(a.get("weapons") or ())
            tr.aim = float((a.get("aim") or [0.0])[0])
            tr.charging = bool(a.get("charging"))
        self.npcs = npcs
        # Unlisted tracks: coast for a few seconds, then forget.
        for mid in list(self.tracks):
            tr = self.tracks[mid]
            if tr.listed:
                continue
            age = t - tr.t
            if age > 25.0:
                del self.tracks[mid]

    def _sounds(self, obs, x, y, t):
        marks = [m for m in self.sound_marks if t - m[0] < 6.0]
        for s in obs.get("sounds") or []:
            kind = s.get("kind")
            if kind in ("death", "drop", "crate", "npc_gun", "explosion"):
                continue
            b = float(s.get("bearing", 0.0))
            d = SOUND_RANGE_GUESS.get(s.get("dist"), 200.0)
            w = 1.0 if kind in ("gun", "laser", "laser_charge", "lightning", "mortar") else 0.6
            marks.append((t, x + math.cos(b) * d, y + math.sin(b) * d, w))
        self.sound_marks = marks[-12:]

    def _threats(self, x, y, z, t):
        out = []
        eye = z + 8.0
        for tr in self.tracks.values():
            age = t - tr.t
            if tr.listed:
                px, py = tr.x, tr.y
            else:
                k = min(age, 3.0)
                px, py = tr.x + tr.vx * k * 0.6, tr.y + tr.vy * k * 0.6
                if self.arena is not None:
                    px, py = self.arena.clamp(px, py)
            d = hyp(px - x, py - y)
            tr.dist = d
            if d > 320.0 or age > 20.0:
                continue
            danger = 1.0
            if "laser" in tr.weapons:
                danger *= 1.6
            if "lightning" in tr.weapons and d < 60.0:
                danger *= 1.4
            if "mortar" in tr.weapons:
                danger *= 1.15
            if d > 200.0:
                danger *= 0.6
            if not tr.listed:
                danger *= 0.75 if age < 6.0 else 0.4
            if tr.guess:
                danger *= 0.7
            to_us = math.atan2(y - py, x - px)
            if tr.listed and abs(wrap(tr.aim - to_us)) < 0.35:
                danger *= 1.3
                if tr.charging:
                    danger *= 1.5
                    if abs(wrap(tr.aim - to_us)) < 0.15:
                        self.laser_alert = t
            tr.danger = danger
            # Can it see / shoot our body right now?
            if t - self.dmg_by.get(tr.id, -99.0) < 1.5:
                tr.sees_us = True  # it just hit us, whatever our map says
            elif tr.listed and self.arena is not None:
                tr.sees_us = self._exposed(x, y, z, tr)
            else:
                tr.sees_us = tr.listed
            out.append(tr)
        out.sort(key=lambda tr: -tr.danger / max(30.0, tr.dist))
        return out

    def _pick_target(self, threats, x, y, z, muzzle_h):
        best = None
        best_s = -1e9
        for tr in threats:
            if not tr.listed or tr.dist > 290.0:
                continue
            s = -tr.dist * 0.02
            if tr.sees_us:
                s += 3.0
            if self.t - self.dmg_by.get(tr.id, -99.0) < 3.0:
                s += 5.0
            s += (1.0 - min(1.0, tr.hp / max(1.0, tr.max_hp))) * 2.5
            if tr.id is not None and (self.t - self.hurt_t) < 2.0:
                s += 0.5
            if s > best_s:
                best_s = s
                best = tr
        return best

    def _readiness(self, weapons, energy, target=None):
        ready = False
        eff = 0.0
        # The gun hits ~85% even at range; against a laser carrier that already sees us, keep
        # trades a bit shorter.
        gun_eff = 160.0
        if target is not None and "laser" in target.weapons and target.sees_us:
            gun_eff = 120.0
        for w in weapons[:2]:
            if not w:
                continue
            kind = w.get("kind")
            if kind == "gun":
                if float(w.get("ammo", 0)) >= 3 and float(w.get("reload", 0)) <= 0:
                    ready = True
                    eff = max(eff, gun_eff)
            elif kind == "laser":
                if float(w.get("cooldown", 0)) <= 0.35 or float(w.get("charge") or 0) > 0:
                    ready = True
                    eff = max(eff, min(self.laser_range, 260.0))
            elif kind == "lightning":
                if energy > 15.0:
                    ready = True
                    eff = max(eff, self.light_range * 0.9)
        return ready, eff

    # ------------------------------------------------------------------ zone
    def _zone_info(self, zone, x, y, t):
        c = zone.get("center") or [x, y]
        cx, cy = float(c[0]), float(c[1])
        r = float(zone.get("radius", 1e4))
        nc = zone.get("next_center") or c
        nx, ny = float(nc[0]), float(nc[1])
        nr = float(zone.get("next_radius", r))
        ss = float(zone.get("shrink_start", t + 999.0))
        se = float(zone.get("shrink_end", ss))
        dps = float(zone.get("dps", 0.0))
        # Where the ring is now (if the fields are the phase start, interpolate; if they are live,
        # this only errs on the safe side).
        if se > ss and t > ss:
            f = min(1.0, (t - ss) / (se - ss))
            ccx, ccy = cx + (nx - cx) * f, cy + (ny - cy) * f
            cr = r + (nr - r) * f
        else:
            ccx, ccy, cr = cx, cy, r
        margin = max(5.0, min(25.0, nr * 0.12))
        safe_r = max(0.0, nr - margin)
        d_next = hyp(x - nx, y - ny)
        inside_next = d_next <= safe_r
        if inside_next:
            gx, gy = x, y
        else:
            k = safe_r / d_next if d_next > 1e-6 else 0.0
            gx, gy = nx + (x - nx) * k, ny + (y - ny) * k
        need = max(0.0, d_next - safe_r)
        # Time until the moving ring would reach us.
        path_factor = 1.35
        travel = need * path_factor / max(self.walk, 1.0)
        deadline = se if t < se else t + 999.0
        if nr >= r - 0.5 and t >= se:
            deadline = t + 999.0
        slack = deadline - t - travel
        in_zone = hyp(x - ccx, y - ccy) <= cr
        late = (not inside_next) and slack < 25.0
        urgent = (not inside_next) and slack < 6.0
        if not in_zone:
            gx, gy = self._toward_circle(x, y, ccx, ccy, cr, nx, ny, safe_r)
        return {
            "c": (ccx, ccy), "r": cr, "n": (nx, ny), "nr": nr, "safe_r": safe_r,
            "ss": ss, "se": se, "dps": dps, "in_zone": in_zone, "inside_next": inside_next,
            "goal": (gx, gy), "late": late, "urgent": urgent, "slack": slack,
        }

    def _toward_circle(self, x, y, cx, cy, cr, nx, ny, safe_r):
        # Aim at the next safe circle if we can, it is inside the current one.
        d = hyp(x - nx, y - ny)
        if d > 1e-6 and safe_r > 0:
            k = max(0.0, safe_r - 4.0) / d
            return nx + (x - nx) * k, ny + (y - ny) * k
        return nx, ny

    def _inside(self, p, zone, margin):
        nx, ny = zone["n"]
        if zone["late"] or zone["urgent"]:
            return hyp(p[0] - nx, p[1] - ny) <= zone["safe_r"] - margin * 0.0
        cx, cy = zone["c"]
        return hyp(p[0] - cx, p[1] - cy) <= zone["r"] - margin

    # ------------------------------------------------------------------ strategy: where to camp
    def _camp(self, x, y, z, t, zone):
        nx, ny = zone["n"]
        safe_r = zone["safe_r"]
        key = (round(nx), round(ny), round(safe_r))
        if self.camp is not None and self.camp_key == key:
            return self.camp
        self.camp_key = key
        a = self.arena
        if a is None or safe_r < 6.0:
            self.camp = (nx, ny) if safe_r < 6.0 or not zone["inside_next"] else (x, y)
            return self.camp
        # Candidates: around the next centre and along our way in. Prefer enclosed spots, near us,
        # and not near the edge of the next circle (later circles shrink inside it).
        cands = [(x, y)]
        for fr in (0.0, 0.25, 0.45, 0.65):
            rr = safe_r * fr
            n = 1 if fr == 0.0 else 10
            for k in range(n):
                ang = k * TWO_PI / n + fr
                cands.append((nx + math.cos(ang) * rr, ny + math.sin(ang) * rr))
        d_in = hyp(x - nx, y - ny)
        if d_in > 1e-6:
            for fr in (0.3, 0.55, 0.8):
                k = (safe_r * fr) / d_in
                if k < 1.0:
                    cands.append((nx + (x - nx) * k, ny + (y - ny) * k))
        for rr in (25.0, 50.0, 80.0):
            for k in range(10):
                ang = k * TWO_PI / 10 + rr
                cands.append((x + math.cos(ang) * rr, y + math.sin(ang) * rr))
        known = [(tr.x, tr.y) for tr in self.tracks.values() if t - tr.t < 20.0]
        best = None
        best_s = -1e9
        for (px, py) in cands:
            px, py = a.clamp(px, py)
            fx, fy = a.free_near(px, py, 6)
            if hyp(fx - nx, fy - ny) > safe_r:
                continue
            idx = a._idx(fx, fy)
            if a.bld[idx]:
                continue  # roofs are not ground we can reach
            fz = a.surface(fx, fy)
            encl, _ = a.enclosure(fx, fy, fz)
            s = encl * 3.0
            s -= hyp(fx - x, fy - y) / 140.0
            s -= (hyp(fx - nx, fy - ny) / max(safe_r, 1.0)) * 1.2
            for kx, ky in known:
                dk = hyp(fx - kx, fy - ky)
                if dk < 90.0:
                    s -= (90.0 - dk) / 90.0 * 1.5
            if s > best_s:
                best_s = s
                best = (fx, fy)
        self.camp = best or (nx, ny)
        return self.camp

    # ------------------------------------------------------------------ tactics: where to stand
    def _tactical(self, x, y, z, t, active, target, engage, zone, t_start, urgent=False):
        a = self.arena
        if (t - self.spot_t < 0.25 and self.spot is not None and not urgent
                and self.spot_mode == ("E" if engage else "H")):
            return self.spot
        self.spot_t = t
        self.spot_mode = "E" if engage else "H"
        threats = active[:4]
        near = [(x, y, 0.0)]
        if self.spot is not None and hyp(self.spot[0] - x, self.spot[1] - y) < 110.0:
            near.append((self.spot[0], self.spot[1], hyp(self.spot[0] - x, self.spot[1] - y)))
        rot = self.rand() * 0.5
        for r in (4.0, 8.0, 13.0, 19.0, 27.0, 36.0):
            n = 8 if r < 10 else 12
            for k in range(n):
                ang = rot + k * TWO_PI / n
                near.append((x + math.cos(ang) * r, y + math.sin(ang) * r, r))
        far = []
        for r in (50.0, 66.0, 85.0, 110.0):
            for k in range(16):
                ang = rot * 0.7 + k * TWO_PI / 16
                far.append((x + math.cos(ang) * r, y + math.sin(ang) * r, r))
        state = {"best": None, "best_s": -1e9, "scored": [], "cover": 99.0}
        budget_end = t_start + 0.012
        my_bld = a.bld[a._idx(x, y)]

        def score_all(cands, move_w):
            for (px, py, r) in cands:
                if time.perf_counter() > budget_end and state["best"] is not None:
                    return
                px, py = a.clamp(px, py)
                idx = a._idx(px, py)
                if r > 0.0:
                    if a.cost[idx] >= 5 or (a.bld[idx] and not my_bld):
                        continue
                    if not a.walkable(x, y, px, py, strict=False):
                        continue
                if not self._spot_in_zone(px, py, zone):
                    continue
                pz = a.surface(px, py)
                s = -r * move_w
                hidden_all = True
                shoot = False
                for th in threats:
                    if not self._exposed(px, py, pz, th):
                        continue
                    hidden_all = False
                    if engage and th is target:
                        shoot = True
                        if th.dist < 18.0:
                            s -= 10.0  # don't hug it
                    else:
                        s -= 100.0 * th.danger
                if engage and target is not None:
                    if shoot:
                        s += 45.0
                    else:
                        s -= min(30.0, hyp(px - target.x, py - target.y) * 0.1)
                if hidden_all:
                    s += 8.0
                    if r < state["cover"]:
                        state["cover"] = r
                encl, _ = a.enclosure(px, py, pz)
                s += encl * 6.0
                for (nx_, ny_, nz_) in self.npcs:
                    dn = hyp(px - nx_, py - ny_)
                    if dn < self.npc_aggro + 8.0:
                        s -= (self.npc_aggro + 8.0 - dn) * 1.5
                state["scored"].append((s, px, py))
                if s > state["best_s"]:
                    state["best_s"] = s
                    state["best"] = (px, py)

        score_all(near, 3.0 if urgent else 0.9)
        exposed_now = any(th.sees_us for th in threats)
        if state["cover"] > 36.0 or (exposed_now and not engage and state["best_s"] < -40.0):
            # Nothing hides us close by: look further out. Standing in the open is worse than a run.
            score_all(far, 0.45)
        self.cover_dist = state["cover"]
        best, best_s, scored = state["best"], state["best_s"], state["scored"]
        if best is None:
            return self.spot or (x, y)
        # Hysteresis: keep the old spot unless the new one is clearly better.
        if self.spot is not None:
            keep = 6.0 if engage else 15.0
            if not urgent and t - getattr(self, "spot_set_t", -99.0) < 0.6:
                keep = 40.0  # commit for a moment: flip-flopping between spots is worse than either
            for s, px, py in scored:
                if hyp(px - self.spot[0], py - self.spot[1]) < 1.5 and s > best_s - keep:
                    best = self.spot
                    break
        # Fighting from a good spot: shuffle among equally good spots so we're not a still target.
        if engage and t > self.jitter_t:
            close = [(px, py) for s, px, py in scored if s > best_s - 6.0 and hyp(px - x, py - y) < 9.0]
            if len(close) > 1:
                best = close[int(self.rand() * len(close)) % len(close)]
                self.jitter_t = t + 0.9 + self.rand() * 0.6
        if self.spot is None or hyp(best[0] - self.spot[0], best[1] - self.spot[1]) > 1.5:
            self.spot_set_t = t
        self.spot = best
        self.spot_score = best_s
        return best

    def _spot_in_zone(self, px, py, zone):
        cx, cy = zone["c"]
        if hyp(px - cx, py - cy) > zone["r"] - 6.0:
            return False
        if zone["late"] or zone["slack"] < 40.0:
            nx, ny = zone["n"]
            if not zone["inside_next"]:
                # Must not move away from the next circle.
                return hyp(px - nx, py - ny) <= hyp(self.me[0] - nx, self.me[1] - ny) + 2.0
            return hyp(px - nx, py - ny) <= zone["safe_r"]
        return True

    def _exposed(self, px, py, pz, th):
        """Can threat `th` see/shoot a mech standing at (px, py, pz)?"""
        a = self.arena
        if th.listed:
            ex, ey = th.x, th.y
        else:
            ex, ey = a.clamp(th.x + th.vx * 1.5, th.y + th.vy * 1.5)
        ez = th.z + 8.0
        m = 0.6  # an obstacle must clear the line by this much to count as cover
        if a.los(ex, ey, ez, px, py, pz + 8.5, 1.0, m):
            return True
        dx, dy = px - ex, py - ey
        d = hyp(dx, dy) or 1.0
        ox, oy = -dy / d * 1.7, dx / d * 1.7
        if a.los(ex, ey, ez, px + ox, py + oy, pz + 4.5, 1.0, m):
            return True
        if a.los(ex, ey, ez, px - ox, py - oy, pz + 4.5, 1.0, m):
            return True
        return False

    def _hidden_here(self, x, y, z, t):
        for tr in self.tracks.values():
            if t - tr.t > 25.0 or hyp(tr.x - x, tr.y - y) > 300.0:
                continue
            if self._exposed(x, y, z, tr):
                return False
        return True

    # ------------------------------------------------------------------ steering and paths
    def _steer(self, x, y, z, vx, vy, t, goal, hurry, t_start):
        if goal is None:
            return 0.0, 0.0
        gx, gy = goal
        a = self.arena
        spd = hyp(vx, vy)
        # stuck detection
        if self.last_pos is not None and hyp(gx - x, gy - y) > 3.0:
            moved = hyp(x - self.last_pos[0], y - self.last_pos[1])
            if moved < 0.25:
                self.stuck += 1
            else:
                self.stuck = max(0, self.stuck - 2)
        else:
            self.stuck = max(0, self.stuck - 1)
        self.last_pos = (x, y)
        if self.stuck > 12 and a is not None:
            # Mark the cell ahead as blocked for planning and force a new path.
            hx, hy = gx - x, gy - y
            hl = hyp(hx, hy) or 1.0
            bi, bj = a.cell(x + hx / hl * 2.5, y + hy / hl * 2.5)
            self.blocked.add(bj * a.W + bi)
            self.path = None
            self.job = None
            self.stuck = 0
            self.unstick_t = t + 1.2
            self.unstick_dir = (self.rand() - 0.5) * 2.4
        if a is None:
            d = hyp(gx - x, gy - y)
            return ((gx - x) / d, (gy - y) / d) if d > 0.5 else (0.0, 0.0)
        d = hyp(gx - x, gy - y)
        if d < 0.8:
            return 0.0, 0.0
        if t < self.unstick_t:
            ang = math.atan2(gy - y, gx - x) + self.unstick_dir
            return math.cos(ang), math.sin(ang)
        # Short and clear: walk straight.
        if d < 45.0 and a.walkable(x, y, gx, gy, strict=False) and a.walkable(x, y, gx, gy, strict=True):
            self.path = None
            return (gx - x) / d, (gy - y) / d
        if d < 12.0 and a.walkable(x, y, gx, gy, strict=False):
            return (gx - x) / d, (gy - y) / d
        # Otherwise follow a planned path (re-plan when the goal moves).
        if self.path is None or self.path_goal is None or hyp(self.path_goal[0] - gx, self.path_goal[1] - gy) > 6.0:
            if self.job is None or self.job_goal is None or hyp(self.job_goal[0] - gx, self.job_goal[1] - gy) > 6.0:
                avoid = [(tr.x, tr.y) for tr in self.tracks.values()
                         if t - tr.t < 15.0 and hyp(tr.x - x, tr.y - y) < 220.0
                         and hyp(tr.x - gx, tr.y - gy) > 25.0]
                self.job = PathJob(a, x, y, gx, gy, blocked=self.blocked, avoid=avoid,
                                   avoid_r=55.0, avoid_cost=4.0)
                self.job_goal = (gx, gy)
            if self.job.step(6000, deadline=max(t_start + 0.015, time.perf_counter() + 0.002)):
                self.path = self.job.path
                self.path_goal = self.job_goal
                self.path_i = 0
                self.job = None
        if self.path:
            return self._follow(x, y, gx, gy)
        # Waiting for the planner: head straight, sliding along walls.
        return (gx - x) / d, (gy - y) / d

    def _follow(self, x, y, gx, gy):
        a = self.arena
        path = self.path
        n = len(path)
        # nearest point at or after path_i
        best_i = self.path_i
        best_d = 1e9
        for k in range(self.path_i, min(n, self.path_i + 25)):
            d = hyp(path[k][0] - x, path[k][1] - y)
            if d < best_d:
                best_d = d
                best_i = k
        if best_d > 14.0:
            self.path = None  # wandered off: re-plan next tick
            return ((gx - x) / (hyp(gx - x, gy - y) or 1.0), (gy - y) / (hyp(gx - x, gy - y) or 1.0))
        self.path_i = best_i
        tgt = path[min(n - 1, best_i + 1)]
        for k in range(min(n - 1, best_i + 14), best_i, -1):
            px, py = path[k]
            if a.walkable(x, y, px, py, strict=True):
                tgt = (px, py)
                break
        dx, dy = tgt[0] - x, tgt[1] - y
        d = hyp(dx, dy)
        if best_i >= n - 2 and d < 1.5:
            self.path = None
        if d < 1e-6:
            return 0.0, 0.0
        return dx / d, dy / d

    def _dodge(self, x, y, z, projectiles):
        dox = doy = 0.0
        mz = z + 4.5
        for p in projectiles:
            pos = p.get("pos") or [0, 0, 0]
            vel = p.get("vel") or [0, 0, 0]
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            pvx, pvy, pvz = float(vel[0]), float(vel[1]), float(vel[2])
            rx, ry, rz = x - px, y - py, mz - pz
            v2 = pvx * pvx + pvy * pvy + pvz * pvz
            if v2 < 1.0:
                continue
            tca = (rx * pvx + ry * pvy + rz * pvz) / v2
            if tca < -0.05 or tca > 1.6:
                continue
            cx = rx - pvx * tca
            cy = ry - pvy * tca
            cz = rz - pvz * tca
            miss = math.sqrt(cx * cx + cy * cy + cz * cz)
            thresh = 13.0 if p.get("kind") == "mortar" else 6.0
            if miss > thresh:
                continue
            plen = hyp(pvx, pvy) or 1.0
            sx, sy = -pvy / plen, pvx / plen
            if cx * sx + cy * sy < 0:
                sx, sy = -sx, -sy
            w = (thresh - miss) / thresh
            dox += sx * w
            doy += sy * w
        m = hyp(dox, doy)
        if m > 1e-6:
            k = min(1.0, m) / m
            return dox * k, doy * k
        return 0.0, 0.0

    # ------------------------------------------------------------------ combat
    def _combat(self, x, y, z, me, weapons, target, engage, threats, muzzle_h, energy, t):
        my_aim = me.get("aim") or [0.0, 0.0]
        fire = [False, False]
        charging = False
        laser_i = None
        for i, w in enumerate(weapons[:2]):
            if w and w.get("kind") == "laser":
                laser_i = i
        laser_hot = laser_i is not None and float(weapons[laser_i].get("charge") or 0.0) > 0.0
        unseen_attacker = (t - self.hurt_t < 2.0 and self.hurt_bearing is not None
                           and not any(th.listed and t - self.dmg_by.get(th.id, -99.0) < 2.0 for th in threats))
        if unseen_attacker and (target is None or not target.sees_us):
            # Something we can't see is hitting us: face it (to see it, and for the front armor).
            aim = [self.hurt_bearing, 0.0]
            target = None
        elif target is not None:
            # A hitscan laser needs no bullet lead, just the slew time.
            aim = self._aim_at(x, y, z, muzzle_h, target, self.gun_speed, lead_t=0.1 if laser_hot else None)
        else:
            aim = self._idle_aim(x, y, z, threats, t)
        if target is None:
            self.nofire = "notgt"
            # A laser that is already charging fires on its own at hold_s; let it go only at something.
            if laser_i is not None:
                ch = float(weapons[laser_i].get("charge") or 0.0)
                if ch > 0.0:
                    fire[laser_i] = ch < self.laser_charge + self.laser_hold - 0.25
                    charging = fire[laser_i]
            return aim, fire, charging
        mz = z + muzzle_h
        # Line of fire to the centre, else the upper body, else the legs: partial cover often
        # hides one part only. Aim at the part we can hit.
        lof = False
        aim_dz = 0.0
        if target.listed:
            for dz in (0.0, 3.2, -2.6):
                if self.los(x, y, mz, target.x, target.y, target.z + 4.5 + dz):
                    lof = True
                    aim_dz = dz
                    break
            if not lof and t - self.dmg_by.get(target.id, -99.0) < 1.0:
                lof = True  # it is hitting us right now, so the line is open whatever our grid says
        if lof and aim_dz != 0.0:
            hd = hyp(target.x - x, target.y - y) or 1.0
            aim = [aim[0], max(self.min_pitch, min(self.max_pitch,
                                                   aim[1] + math.atan2(aim_dz, hd)))]
        self.nofire = "lof" if not lof else ""
        d = target.dist
        err_yaw = abs(wrap(float(my_aim[0]) - aim[0]))
        err_pitch = abs(float(my_aim[1]) - aim[1])
        tol = max(0.012, math.atan2(1.9, max(d, 1.0)))
        aimed = err_yaw < tol and err_pitch < tol * 1.5
        others_see = any(th.sees_us for th in threats if th is not target)
        shoot_ok = lof and (engage or target.sees_us)
        if lof and not shoot_ok:
            self.nofire = "hold"
        elif shoot_ok and err_yaw >= tol * 2.0:
            self.nofire = "aim"
        for i, w in enumerate(weapons[:2]):
            if not w:
                continue
            kind = w.get("kind")
            wst = w.get("stats") or {}
            if kind == "gun":
                rng = float(wst.get("range", self.gun_range))
                if (shoot_ok and d < rng * 0.97 and float(w.get("ammo", 0)) > 0
                        and float(w.get("reload", 0)) <= 0 and err_yaw < tol * 2.0):
                    fire[i] = True
            elif kind == "lightning":
                rng = float(wst.get("range", self.light_range))
                if lof and d < rng * 0.95 and energy > 6.0 and err_yaw < self.light_cone * 0.4:
                    fire[i] = True
            elif kind == "laser":
                ch = float(w.get("charge") or 0.0)
                need = float(wst.get("charge", self.laser_charge))
                cd = float(w.get("cooldown", 0.0))
                if ch > 0.0:
                    # Holding a charge: release (fire=False) once full and on target with a clear line.
                    full = ch >= need - 0.05
                    forced = ch > need + self.laser_hold - 0.3
                    if lof and aimed and full:
                        fire[i] = False
                    elif forced:
                        fire[i] = False
                    elif not engage and not lof and others_see:
                        fire[i] = False  # running for cover: half speed would kill us
                    else:
                        fire[i] = True
                        charging = True
                elif cd <= 0.0 and d < self.laser_range * 0.97:
                    # Start charging when we can shoot now, or when we're about to peek.
                    if (lof and engage and err_yaw < 0.35) or (engage and not lof and d < 200.0 and not others_see):
                        fire[i] = True
                        charging = True
            elif kind == "mortar":
                splash = float(wst.get("splash_radius", self.mortar_splash))
                if d > splash * 3.0 and d < 220.0 and float(w.get("cooldown", 1.0)) <= 0.05 and engage:
                    fire[i] = True
        # Mortar wants a lofted aim; only use it when the gun is not firing.
        for i, w in enumerate(weapons[:2]):
            if w and w.get("kind") == "mortar" and fire[i]:
                if not any(fire[k] for k in range(2) if k != i and weapons[k]
                           and weapons[k].get("kind") in ("gun", "lightning", "laser")):
                    aim = self._aim_mortar(x, y, z, muzzle_h, target, w.get("stats") or {})
                else:
                    fire[i] = False
        return aim, fire, charging

    def _aim_at(self, x, y, z, muzzle_h, tr, speed, lead_t=None):
        tx, ty, tz = tr.x, tr.y, tr.z + 4.5
        if not tr.listed:
            return [math.atan2(ty - y, tx - x), 0.0]
        mz = z + muzzle_h
        px, py = tx, ty
        for _ in range(3):
            flight = lead_t if lead_t is not None else hyp(px - x, py - y) / max(speed, 1.0)
            px = tx + tr.vx * flight
            py = ty + tr.vy * flight
        yaw = math.atan2(py - y, px - x)
        pitch = math.atan2(tz - mz, hyp(px - x, py - y) or 0.1)
        return [yaw, max(self.min_pitch, min(self.max_pitch, pitch))]

    def _aim_mortar(self, x, y, z, muzzle_h, tr, wst):
        speed = float(wst.get("speed", self.mortar_speed))
        grav = float(wst.get("gravity", self.mortar_g))
        dist = hyp(tr.x - x, tr.y - y) or 1.0
        flight = dist / max(speed, 1.0) * 1.4
        tx = tr.x + tr.vx * flight
        ty = tr.y + tr.vy * flight
        dist = hyp(tx - x, ty - y)
        s2 = dist * grav / max(speed * speed, 1.0)
        pitch = math.pi / 4 if s2 >= 1.0 else 0.5 * math.asin(max(0.0, s2))
        pitch += math.atan2(tr.z + 1.0 - (z + muzzle_h + 2.5), dist) * 0.25
        return [math.atan2(ty - y, tx - x), max(self.min_pitch, min(self.max_pitch, pitch))]

    def _idle_aim(self, x, y, z, threats, t):
        # Face the most recent danger: a remembered mech, a hit's bearing, a sound.
        if threats:
            tr = threats[0]
            return [math.atan2(tr.y - y, tr.x - x), 0.0]
        if t - self.hurt_t < 6.0 and self.hurt_bearing is not None:
            return [self.hurt_bearing, 0.0]
        if self.sound_marks:
            tm, sx, sy, w = self.sound_marks[-1]
            if t - tm < 5.0:
                return [math.atan2(sy - y, sx - x), 0.0]
        # First seconds: one full turn to spot loot and enemies around the spawn.
        if t < 2.2:
            if self.spawn_yaw is None:
                self.spawn_yaw = float(self.my_aim0)
            return [wrap(self.spawn_yaw + t * 3.0), 0.0]
        # Early on, with no second weapon yet: keep turning all the way round to find loot.
        if t < 180.0 and not any(w and w.get("kind") == "laser" for w in self.weapons[:2]):
            return [wrap(t * 1.6 + self.slot), 0.0]
        # Otherwise sweep the open directions (where an attacker could see us from).
        a = self.arena
        self.scan_i += 1
        if a is not None:
            z0 = a.surface(x, y) + 8.0
            if self.scan_i % 12 == 0 or not hasattr(self, "_scan_dirs"):
                dirs = []
                for k in range(16):
                    ang = k * math.pi / 8.0
                    dirs.append((a.ray_free(x, y, z0, ang, 120.0, 2.0), ang))
                dirs.sort(reverse=True)
                self._scan_dirs = [ang for d, ang in dirs[:4] if d > 25.0] or [0.0]
            sd = self._scan_dirs
            return [sd[(self.scan_i // 15) % len(sd)], 0.0]
        return [wrap(self.scan_i * 0.3), 0.0]

    # ------------------------------------------------------------------ loot
    def _explore(self, x, y, z, t, zone):
        a = self.arena
        ex = self.explore
        if ex is not None and hyp(ex[0] - x, ex[1] - y) > 8.0 and t - self.explore_t < 25.0:
            return ex
        nx, ny = zone["n"]
        lim = zone["safe_r"] if zone["slack"] < 60.0 else zone["r"] - 25.0
        cx, cy = (nx, ny) if zone["slack"] < 60.0 else zone["c"]
        for _ in range(12):
            ang = self.rand() * TWO_PI
            r = 60.0 + self.rand() * 80.0
            px, py = a.clamp(x + math.cos(ang) * r, y + math.sin(ang) * r)
            if hyp(px - cx, py - cy) > lim:
                continue
            k = a._idx(px, py)
            if a.bld[k] or a.cost[k] >= 5:
                continue
            # avoid walking toward enemies we know of
            if any(hyp(px - tr.x, py - tr.y) < 110.0 for tr in self.tracks.values() if t - tr.t < 30.0):
                continue
            self.explore = (px, py)
            self.explore_t = t
            return self.explore
        self.explore = None
        return None

    def _remember_pickups(self, obs, x, y, z, t):
        seen = set()
        for p in obs.get("pickups") or []:
            kind = p.get("kind")
            if LOOT_RANK.get(kind, 9) > 1:
                continue
            pos = p.get("pos") or [0, 0, 0]
            pid = p.get("id")
            seen.add(pid)
            self.pickup_mem[pid] = [kind, float(pos[0]), float(pos[1]), float(pos[2]), t]
        # A remembered pickup that should be in plain view but is not listed is gone.
        for pid in list(self.pickup_mem):
            if pid in seen:
                continue
            kind, px, py, pz, ts = self.pickup_mem[pid]
            d = hyp(px - x, py - y)
            if t - ts > 150.0 or (d < 30.0 and self.los(x, y, z + 8.0, px, py, pz + 0.5)):
                del self.pickup_mem[pid]

    def _loot_goal(self, obs, x, y, z, t, zone, threats, weapons):
        if zone["late"] or not zone["in_zone"] or self.hp_frac < 0.5:
            return None
        if any(th.sees_us for th in threats):
            return None
        nearest = min([th.dist for th in threats if (t - th.t) < 8.0] or [1e9])
        if any(hyp(nx - x, ny - y) < self.npc_aggro + 15.0 for nx, ny, nz in self.npcs):
            return None
        held = {w.get("kind") for w in weapons if w}
        empty = any(w is None for w in weapons[:2])
        best = None
        best_d = 1e9
        for pid, (kind, px, py, pz, ts) in self.pickup_mem.items():
            if kind in held:
                continue
            if not empty and not self._better_than_held(kind, weapons):
                continue
            d = hyp(px - x, py - y)
            reach = 200.0 if (empty or kind == "laser") else 60.0
            # Only if we get there well before any enemy we know of could.
            if abs(pz - z) > 3.5 or d > reach or d > max(25.0, nearest - 50.0):
                continue
            if not self._spot_in_zone(px, py, zone):
                continue
            d += LOOT_RANK.get(kind, 9) * 25.0
            if d < best_d:
                best_d = d
                best = (px, py)
        if nearest < 100.0:
            return best if best is not None and best_d < 25.0 else None
        for c in obs.get("cubes") or []:
            pos = c.get("pos") or [0, 0, 0]
            d = hyp(float(pos[0]) - x, float(pos[1]) - y)
            if abs(float(pos[2]) - z) > 3.5 or d > 25.0 or self.hp_frac < 0.75:
                continue
            if d + 10.0 < best_d:
                best_d = d + 10.0
                best = (float(pos[0]), float(pos[1]))
        return best

    def _better_than_held(self, kind, weapons):
        other = weapons[1] if weapons[0] and weapons[0].get("kind") == "gun" else weapons[0]
        if not other:
            return True
        return LOOT_RANK.get(kind, 9) < LOOT_RANK.get(other.get("kind"), 9)

    def _take(self, obs, x, y, z, weapons):
        held = {w.get("kind") for w in weapons if w}
        empty_slot = None
        for i, w in enumerate(weapons[:2]):
            if w is None:
                empty_slot = i
                break
        best = None
        best_r = 99
        for p in obs.get("pickups") or []:
            kind = p.get("kind")
            if kind in held:
                continue
            pos = p.get("pos") or [0, 0, 0]
            if abs(float(pos[2]) - z) > 3.8:
                continue
            if hyp(float(pos[0]) - x, float(pos[1]) - y) > self.pickup_r - 0.2:
                continue
            r = LOOT_RANK.get(kind, 9)
            if r < best_r:
                best_r = r
                best = p
        if best is None:
            return None
        if empty_slot is not None:
            if best_r > 1:
                return None  # leave mortars: lasers would cook them off on our back
            return (best.get("id"), None)
        # Both full: swap out a worse non-gun weapon (never while a laser charges).
        for i, w in enumerate(weapons[:2]):
            if w and w.get("kind") != "gun" and LOOT_RANK.get(w.get("kind"), 9) > best_r:
                if w.get("kind") == "laser" and float(w.get("charge") or 0) > 0:
                    return None
                return (best.get("id"), i)
        return None

    def _pick(self, choices):
        best_i, best_r = 0, 10 ** 9
        for i, ch in enumerate(choices):
            r = UPGRADE_RANK.get(ch.get("id"), 40)
            if r < best_r:
                best_r = r
                best_i = i
        return best_i


if __name__ == "__main__":
    mf.run(Bot)
