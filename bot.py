"""Mech Royale bot (team "Just for the fun of it").
Hide in low-exposure cells inside the next zone, duel hunters one at a time from cover,
retreat out of sight to regenerate, collect cubes and a laser/lightning, and face threats."""
import math, os, random, time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.ndimage import distance_transform_edt
import mechfight as mf
from world import World

DEBUG = bool(os.environ.get("MB_DEBUG"))
HERE = os.path.dirname(os.path.abspath(__file__))
TAU = 2 * math.pi
UPGRADE_PRIO = ["plating", "armor", "nanites", "gun_rate", "gun_damage", "laser_charge", "laser_damage",
                "lightning_efficiency", "servos", "gyros", "gun_mag", "gun_velocity", "lightning_range", "sensors",
                "jets", "capacitor", "reactor", "lightning_chain", "laser_pierce", "mortar_reload", "mortar_velocity", "mortar_splash"]
WEAPON_PREF = {"laser": 3, "lightning": 2, "mortar": -1, "gun": 0}
P = {
    "expo_w": 60.0, "dist_w": 0.10, "edge_w": 40.0, "enemy_w": 40.0, "seen_w": 150.0, "stick": 10.0,
    "replan_s": 2.5, "path_expo": 1.5, "fight_hp": 0.45, "retreat_hp": 0.30, "band_lo": 55.0, "band_hi": 140.0,
    "engage_far": 130.0, "loot_r": 60.0, "cube_r": 70.0, "outnum": 3,
}
for k in list(P):
    v = os.environ.get("MB_" + k.upper())
    if v is not None:
        P[k] = float(v)


def angdiff(a, b):
    return (a - b + math.pi) % TAU - math.pi


class Bot:
    name = os.environ.get("MB_NAME", "just-for-fun")

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.rng = random.Random(hello.get("seed", 0))
        self.W = World(os.path.join(HERE, "maps.npz"))
        self.reach = self.cfg["loot"]["pickup_radius"]
        nb = self.W.navblk; s = self.W.navsurf
        H, Wd = nb.shape; self.H, self.Wd = H, Wd
        e = self.W.expo
        ex = np.ones((H, Wd), np.float32)
        e2 = np.repeat(np.repeat(e, 2, 0), 2, 1); hh, ww = min(H, e2.shape[0]), min(Wd, e2.shape[1])
        ex[:hh, :ww] = e2[:hh, :ww]
        self.expo2 = ex.ravel()
        idx = np.arange(H * Wd).reshape(H, Wd)
        rows, cols, ws = [], [], []
        for dj, di, c in [(0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)]:
            j0, j1 = 0, H - dj; i0, i1 = max(0, -di), Wd - max(0, di)
            A = idx[j0:j1, i0:i1]; B = idx[j0 + dj:j1 + dj, i0 + di:i1 + di]
            ok = ~nb[j0:j1, i0:i1] & ~nb[j0 + dj:j1 + dj, i0 + di:i1 + di] & (
                np.abs(s[j0:j1, i0:i1] - s[j0 + dj:j1 + dj, i0 + di:i1 + di]) < 2.2 * c)
            a_, b_ = A[ok], B[ok]
            wgt = 2.0 * c * (1 + P["path_expo"] * 0.5 * (self.expo2[a_] + self.expo2[b_]))
            rows.append(a_); cols.append(b_); ws.append(wgt)
        r = np.concatenate(rows + cols); cc = np.concatenate(cols + rows); w = np.concatenate(ws + ws)
        self.graph = csr_matrix((w, (r, cc)), shape=(H * Wd, H * Wd))
        jj, ii = np.mgrid[0:H, 0:Wd]
        self.cx = (self.W.x0 + 2 * ii + 0.5).ravel(); self.cy = (self.W.y0 + 2 * jj + 0.5).ravel()
        self.blk = nb.ravel()
        # main connected component + nearest main node for escapes from pockets (stadium bowl, courtyards)
        _, lab = connected_components(self.graph, directed=False)
        lab = lab.copy(); lab[self.blk] = -1
        big = np.bincount(lab[lab >= 0]).argmax()
        self.main = (lab == big)
        _, ind = distance_transform_edt(~self.main.reshape(H, Wd), return_indices=True)
        self.near_main = (ind[0] * Wd + ind[1]).ravel()
        self.goal = None; self.path = []; self.plan_t = -99; self.plan_threat_ids = ()
        self.attackers = {}; self.dmg_bearings = []
        self.stuck_t = 0.0; self.prev_pos = None; self.sidestep = None
        self.scan_phase = self.rng.random() * TAU
        self.strafe_dir = 1; self.strafe_until = 0.0
        self.mode = "hide"; self.last_seen = {}  # id -> (pos, t)
        self.kill_spots = []
        self.timing = []

    # ---------- navigation ----------
    def node(self, x, y):
        i = min(max(int((x - self.W.x0) // 2), 0), self.Wd - 1); j = min(max(int((y - self.W.y0) // 2), 0), self.H - 1)
        n = j * self.Wd + i
        if self.blk[n]:
            best = None
            for r in range(1, 5):
                for dj in range(-r, r + 1):
                    for di in range(-r, r + 1):
                        jj, ii = j + dj, i + di
                        if 0 <= jj < self.H and 0 <= ii < self.Wd and not self.blk[jj * self.Wd + ii]:
                            d = dj * dj + di * di
                            if best is None or d < best[0]:
                                best = (d, jj * self.Wd + ii)
                if best:
                    return best[1]
        return n

    def free_at(self, x, y):
        i = int((x - self.W.x0) // 2); j = int((y - self.W.y0) // 2)
        if not (0 <= i < self.Wd and 0 <= j < self.H):
            return False
        return not self.blk[j * self.Wd + i]

    def plan(self, obs, me, threats, want=None):
        """Pick a goal cell (or use `want` node) and a path to it."""
        t = obs["t"]; z = obs["zone"]
        x, y = me["pos"][0], me["pos"][1]
        src = self.node(x, y)
        dist, pred = dijkstra(self.graph, indices=src, return_predecessors=True, limit=1500)
        reach = np.isfinite(dist)
        if want is not None and reach[want]:
            best = want
        else:
            ncx, ncy, nr = z["next_center"][0], z["next_center"][1], z["next_radius"]
            ccx, ccy, cr = z["center"][0], z["center"][1], z["radius"]
            tgt_r = max(nr, 10.0)
            dn = np.hypot(self.cx - ncx, self.cy - ncy)
            rel = dn / tgt_r
            ok = reach & (rel < 0.9)
            if ok.sum() < 20:
                ok = reach & (dn < tgt_r + 15)
            if ok.sum() < 5:
                ok = reach
            sel = np.flatnonzero(ok)
            sc = P["expo_w"] * self.expo2[sel] + P["dist_w"] * dist[sel] + P["edge_w"] * np.maximum(rel[sel] - 0.5, 0) ** 2 * 4
            if nr < 80:
                sc = sc + 0.6 * dn[sel]
            dc = np.hypot(self.cx[sel] - ccx, self.cy[sel] - ccy)
            sc = sc + np.where(dc > cr - 8, 300.0, 0.0)
            for e in threats:
                de = np.hypot(self.cx[sel] - e["pos"][0], self.cy[sel] - e["pos"][1])
                sc = sc + P["enemy_w"] * np.clip(1 - de / 150.0, 0, 1)
            if self.goal is not None:
                sc = sc - np.where(sel == self.goal, P["stick"], 0.0)
            order = np.argsort(sc)
            if threats:
                cand = sel[order[:150]]; csc = sc[order[:150]].copy()
                feet = np.stack([self.cx[cand], self.cy[cand], self.W.surf(self.cx[cand], self.cy[cand])], 1)
                for e in threats[:4]:
                    ex, ey, ez = e["pos"]
                    vis = self.W.sees(np.tile([ex, ey, ez + 8], (len(cand), 1)), feet, canopy=True)
                    csc += vis * P["seen_w"]
                best = int(cand[np.argmin(csc)])
            else:
                best = int(sel[order[0]])
        self.goal = best
        path = []; n = best; k = 0
        while n != src and n >= 0 and k < 3000:
            path.append(n); n = pred[n]; k += 1
        path.reverse()
        self.path = path
        self.plan_t = t
        if DEBUG:
            mf.log(f"t={t:.1f} plan mode={self.mode} pos=({x:.0f},{y:.0f}) goal=({self.cx[best]:.0f},{self.cy[best]:.0f}) len={len(path)} threats={[e['id'] for e in threats]}")

    def steer(self, x, y):
        while self.path and math.hypot(self.cx[self.path[0]] - x, self.cy[self.path[0]] - y) < 2.5:
            self.path.pop(0)
        if not self.path:
            return None
        n0 = self.path[0]
        if math.hypot(self.cx[n0] - x, self.cy[n0] - y) > 6:
            n = n0
        else:
            n = self.path[min(3, len(self.path) - 1)]
        return math.atan2(self.cy[n] - y, self.cx[n] - x)

    def safe_dir(self, x, y, yaw, look=5.0):
        """Rotate a desired move direction away from walls (nav-blocked cells)."""
        for off in (0, 0.4, -0.4, 0.8, -0.8, 1.3, -1.3, 1.9, -1.9, 2.6, -2.6, math.pi):
            a = yaw + off
            if self.free_at(x + look * math.cos(a), y + look * math.sin(a)) and self.free_at(x + 2.5 * math.cos(a), y + 2.5 * math.sin(a)):
                return a
        return yaw

    # ---------- main ----------
    def act(self, obs):
        t0 = time.perf_counter()
        me = obs["self"]; t = obs["t"]
        x, y, zf = me["pos"]
        st = me["stats"]
        act = {"tick": obs["tick"]}
        hpf = me["hp"] / max(me["max_hp"], 1)
        if obs.get("offer"):
            ch = [c["id"] for c in obs["offer"]["choices"]]
            act["pick"] = min(range(len(ch)), key=lambda i: UPGRADE_PRIO.index(ch[i]) if ch[i] in UPGRADE_PRIO else 99)
        hurt = False
        for ev in obs.get("events", []):
            et = ev["type"]
            if et == "damaged" and ev.get("by") is not None:
                self.attackers[ev["by"]] = t; hurt = True
                if ev.get("bearing") is not None:
                    self.dmg_bearings.append((t, ev["bearing"]))
            elif et == "kill":
                lp = self.last_seen.get(ev["target"])
                if lp: self.kill_spots.append((lp[0][0], lp[0][1], t))
        self.dmg_bearings = [(tt, b) for tt, b in self.dmg_bearings if t - tt < 3]
        eye = [x, y, zf + st.get("eye_height", 8)]
        mz = zf + st.get("muzzle_height", 7)
        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        for e in enemies:
            e["_d"] = math.dist(e["pos"][:2], (x, y))
            self.last_seen[e["id"]] = (e["pos"], t)
        if enemies:
            feet = np.array([e["pos"] for e in enemies], np.float64)
            vis = self.W.sees(np.tile(eye, (len(enemies), 1)), feet, canopy=False)
            for e, v in zip(enemies, vis):
                e["_los"] = bool(v)
                b = math.atan2(y - e["pos"][1], x - e["pos"][0])
                e["_facing"] = abs(angdiff(e["aim"][0], b)) < 0.5
        enemies.sort(key=lambda e: e["_d"])
        threats = [e for e in enemies if e["_d"] < 240]
        vis_threats = [e for e in threats if e["_los"]]
        recent_att = {k for k, v in self.attackers.items() if t - v < 4}
        # ---------- mode ----------
        prev_mode = self.mode
        hostile = [e for e in vis_threats if e["_d"] < P["engage_far"] or e["id"] in recent_att or e["_facing"]]
        if hostile:
            weakest = min(hostile, key=lambda e: e.get("hp", 1400))
            strong_n = len(hostile)
            if hpf < P["retreat_hp"] and weakest.get("hp", 1400) > me["hp"] * 0.8:
                self.mode = "retreat"
            elif strong_n >= P["outnum"] and hpf < 0.85:
                self.mode = "retreat"
            elif hpf >= P["fight_hp"] or weakest.get("hp", 1400) < me["hp"]:
                self.mode = "fight"
            else:
                self.mode = "retreat"
        elif hurt or (recent_att and t - max(self.attackers.values()) < 2):
            self.mode = "retreat"
        else:
            self.mode = "hide"
        # ---------- planning ----------
        tids = tuple(sorted(e["id"] for e in threats[:4]))
        need = (t - self.plan_t > P["replan_s"]) or not self.path
        if self.mode != prev_mode or (tids != self.plan_threat_ids and t - self.plan_t > 0.8):
            need = True
        want = None
        if self.mode == "hide" and not threats:
            want = self.loot_target(obs, me)
        cur_node = self.node(x, y)
        pocket = not self.main[cur_node]
        if need and not pocket:
            try:
                ths = threats if self.mode != "hide" else threats
                if not ths and self.dmg_bearings:
                    b = self.dmg_bearings[-1][1]
                    ths = [{"id": -1, "pos": [x + 80 * math.cos(b), y + 80 * math.sin(b), zf]}]
                self.plan(obs, me, ths, want)
                self.plan_threat_ids = tids
            except Exception as ex:
                mf.log("plan fail", repr(ex))
        move_yaw = None; speed = 1.0; boost = False
        z = obs["zone"]
        dcen = math.hypot(x - z["center"][0], y - z["center"][1])
        zone_urgent = (not me["in_zone"]) or dcen > z["radius"] - 6
        if pocket:
            n = self.near_main[cur_node]
            tx, ty = self.cx[n], self.cy[n]
            move_yaw = math.atan2(ty - y, tx - x)
            if me["grounded"]:
                act["jump"] = True
            boost = me["energy"] > 3
        else:
            move_yaw = self.steer(x, y)
            if move_yaw is None and self.goal is not None and math.hypot(self.cx[self.goal] - x, self.cy[self.goal] - y) < 2.5:
                speed = 0.0
        # ---------- target selection ----------
        target = None
        if vis_threats:
            def prio(e):
                p = e["_d"]
                if e["id"] in recent_att: p -= 80
                p -= (1400 - e.get("hp", 1400)) * 0.12
                if e["_facing"]: p -= 30
                return p
            cands = vis_threats if self.mode != "hide" else hostile
            if cands:
                target = min(cands, key=prio)
        # ---------- fight movement ----------
        if self.mode == "fight" and target is not None and not zone_urgent and not pocket:
            b = math.atan2(target["pos"][1] - y, target["pos"][0] - x)
            if t > self.strafe_until:
                self.strafe_dir = -self.strafe_dir if self.rng.random() < 0.7 else self.strafe_dir
                self.strafe_until = t + self.rng.uniform(0.35, 0.9)
            lo, hi = P["band_lo"], P["band_hi"]
            mine = {w["kind"] for w in me["weapons"] if w}
            theirs = set(target.get("weapons", []))
            if "lightning" in mine and "lightning" not in theirs and target.get("hp", 1400) < me["hp"]:
                lo, hi = 15, 35
            elif "lightning" in theirs:
                lo, hi = max(lo, 60), max(hi, 150)
            radial = 0.0
            if target["_d"] < lo: radial = -0.8
            elif target["_d"] > hi: radial = 0.7
            sx = math.cos(b + math.pi / 2 * self.strafe_dir); sy = math.sin(b + math.pi / 2 * self.strafe_dir)
            mvx = sx + radial * math.cos(b); mvy = sy + radial * math.sin(b)
            # stay inside zone
            if dcen > z["radius"] - 25:
                cz = math.atan2(z["center"][1] - y, z["center"][0] - x); mvx += math.cos(cz); mvy += math.sin(cz)
            move_yaw = self.safe_dir(x, y, math.atan2(mvy, mvx))
            speed = 1.0
            boost = me["energy"] > 40 and target["_d"] < 90
        elif self.mode == "retreat":
            boost = me["energy"] > 10
        if zone_urgent:
            boost = me["energy"] > 5
            if move_yaw is None:
                move_yaw = math.atan2(z["next_center"][1] - y, z["next_center"][0] - x)
        # ---------- aim & fire ----------
        weapons = me["weapons"]
        fire = [False, False]
        aim_yaw = None; aim_pitch = 0.0
        if target is not None:
            aim_yaw, aim_pitch = self.lead(me, target, mz)
            err = abs(angdiff(me["aim"][0], aim_yaw))
            tol = math.atan2(2.0, max(target["_d"], 1)) + 0.006
            shoot = self.mode != "hide" or target in hostile
            los_m = self.W.los([x, y, mz], [target["pos"][0], target["pos"][1], target["pos"][2] + 4.5], canopy=False)[0]
            for si, wp in enumerate(weapons):
                if wp is None or not shoot:
                    continue
                k = wp["kind"]
                if k == "gun":
                    if los_m and err < tol and wp.get("reload", 0) == 0 and target["_d"] < wp["stats"]["range"] * 0.97:
                        fire[si] = True
                elif k == "laser":
                    ch = wp.get("charge", 0) or 0
                    full = wp["stats"]["charge"]
                    if target["_d"] < wp["stats"]["range"]:
                        if ch == 0:
                            fire[si] = wp["cooldown"] <= 0.05
                        elif ch < full * 0.97:
                            fire[si] = True
                        elif (err < tol and los_m) or ch > full + wp["stats"].get("hold_s", 2) - 0.25:
                            fire[si] = False
                        else:
                            fire[si] = True
                elif k == "lightning":
                    if target["_d"] < wp["stats"]["range"] * 0.95 and err < math.radians(wp["stats"]["cone_deg"]) * 0.4 and me["energy"] > 8:
                        fire[si] = True
                elif k == "mortar":
                    pass
        else:
            for si, wp in enumerate(weapons):
                if wp and wp["kind"] == "laser" and (wp.get("charge", 0) or 0) > 0:
                    fire[si] = False
            if self.dmg_bearings:
                aim_yaw = self.dmg_bearings[-1][1]
            elif threats:
                e = threats[0]; aim_yaw = math.atan2(e["pos"][1] - y, e["pos"][0] - x)
            else:
                snd = [s for s in obs.get("sounds", []) if s["kind"] in ("gun", "laser", "laser_charge", "lightning", "step", "boost", "jet") and s["dist"] != "far"]
                if snd:
                    aim_yaw = snd[-1]["bearing"]
                else:
                    self.scan_phase += 0.1
                    base = move_yaw if (move_yaw is not None and speed > 0) else self.scan_phase * 2.2
                    aim_yaw = base + (1.0 * math.sin(self.scan_phase * 3) if speed > 0 else 0)
            g = weapons[0]
            if g and g["kind"] == "gun" and g.get("ammo", 12) < g["stats"]["mag"] * 0.6 and g.get("reload", 0) == 0 and not vis_threats:
                act["reload"] = True
        g = weapons[0]
        if g and g["kind"] == "gun" and g.get("ammo", 1) == 0 and g.get("reload", 0) == 0:
            act["reload"] = True
        # ---------- dodge ----------
        dodge = None
        for e in vis_threats:
            if e.get("charging") and e["_facing"] and abs(angdiff(e["aim"][0], math.atan2(y - e["pos"][1], x - e["pos"][0]))) < 0.12:
                dodge = math.atan2(y - e["pos"][1], x - e["pos"][0]) + math.pi / 2 * self.strafe_dir
        for p in obs.get("projectiles", []):
            if p["kind"] not in ("gun", "npc"):
                continue
            px, py, pz = p["pos"]; vx, vy, vz = p["vel"]
            v2 = vx * vx + vy * vy
            if v2 < 1:
                continue
            tt = ((x - px) * vx + (y - py) * vy) / v2
            if 0.12 < tt < 0.9:
                cxp, cyp = px + vx * tt, py + vy * tt
                if math.hypot(cxp - x, cyp - y) < 3.5:
                    side = 1 if ((cxp - x) * -vy + (cyp - y) * vx) < 0 else -1
                    dodge = math.atan2(vy, vx) + math.pi / 2 * side
        if dodge is not None and self.mode != "hide" and not pocket:
            move_yaw = self.safe_dir(x, y, dodge); speed = 1.0
        # ---------- stuck handling ----------
        if self.prev_pos is not None and move_yaw is not None and speed > 0.5 and not pocket:
            if math.hypot(x - self.prev_pos[0], y - self.prev_pos[1]) < 0.2:
                self.stuck_t += 0.1
            else:
                self.stuck_t = max(0.0, self.stuck_t - 0.2)
        self.prev_pos = (x, y)
        if self.stuck_t > 0.6:
            if me["grounded"]:
                act["jump"] = True
            else:
                boost = me["energy"] > 5
            if self.stuck_t > 1.5 and (self.sidestep is None or t > self.sidestep[1]):
                self.sidestep = (self.rng.random() * TAU, t + 0.7)
            if self.stuck_t > 3.0:
                self.path = []; self.stuck_t = 0.0
        if self.sidestep and t < self.sidestep[1]:
            move_yaw = self.sidestep[0]; speed = 1.0
        self.loot(obs, me, act)
        mv = [0.0, 0.0] if move_yaw is None else [math.cos(move_yaw) * speed, math.sin(move_yaw) * speed]
        act["move"] = mv
        act["boost"] = bool(boost and (speed > 0 or not me["grounded"]))
        if aim_yaw is None:
            aim_yaw = me["aim"][0]
        act["aim"] = [((aim_yaw + math.pi) % TAU) - math.pi, max(-0.65, min(1.2, aim_pitch))]
        act["fire"] = fire
        if DEBUG:
            mf.draw.label(f"{self.mode} hp{me['hp']:.0f}")
        return act

    def lead(self, me, e, mz):
        x, y, _ = me["pos"]
        tx, ty, tz = e["pos"]; vx, vy, vz = e.get("vel", [0, 0, 0])
        spd = 252.0
        hitscan = False
        for wp in me["weapons"]:
            if wp and wp["kind"] == "gun":
                spd = wp["stats"]["speed"]
        w1 = me["weapons"][1]
        if w1 and w1["kind"] == "laser" and (w1.get("charge", 0) or 0) > w1["stats"]["charge"] * 0.6:
            hitscan = True
        tt = 0.0
        if not hitscan:
            for _ in range(3):
                px, py = tx + vx * tt, ty + vy * tt
                tt = math.hypot(px - x, py - y) / spd + 0.03
        px, py, pz = tx + vx * tt, ty + vy * tt, tz + 4.5 + vz * tt
        d = math.hypot(px - x, py - y)
        return math.atan2(py - y, px - x), math.atan2(pz - mz, max(d, 1))

    def loot_target(self, obs, me):
        """Node of a worthwhile nearby pickup/cube inside the safe zone, or None."""
        x, y, zf = me["pos"]
        z = obs["zone"]
        w = me["weapons"]
        best = None
        have = {wp["kind"] for wp in w if wp}
        for p in obs.get("pickups", []):
            k = p["kind"]
            if k in have or WEAPON_PREF.get(k, -1) < 1:
                continue
            if w[1] is not None and WEAPON_PREF.get(k, 0) <= WEAPON_PREF.get(w[1]["kind"], 0):
                continue
            if abs(p["pos"][2] - self.W.surf(p["pos"][0], p["pos"][1])) > 2:
                continue  # on a roof
            d = math.dist(p["pos"][:2], (x, y))
            if d < P["loot_r"] and math.hypot(p["pos"][0] - z["next_center"][0], p["pos"][1] - z["next_center"][1]) < z["next_radius"] + 30:
                sc = d - 20 * WEAPON_PREF[k]
                if best is None or sc < best[0]:
                    best = (sc, p["pos"])
        for c in obs.get("cubes", []):
            d = math.dist(c["pos"][:2], (x, y))
            if d < P["cube_r"] and abs(c["pos"][2] - self.W.surf(c["pos"][0], c["pos"][1])) < 2:
                sc = d - 25
                if best is None or sc < best[0]:
                    best = (sc, c["pos"])
        if best is None:
            return None
        n = self.node(best[1][0], best[1][1])
        return n

    def loot(self, obs, me, act):
        x, y, z = me["pos"]
        w = me["weapons"]
        have = {wp["kind"] for wp in w if wp}
        for p in obs.get("pickups", []):
            if math.dist(p["pos"][:2], (x, y)) > self.reach - 0.3 or abs(p["pos"][2] - z) > 3.8:
                continue
            k = p["kind"]
            if k in have or WEAPON_PREF.get(k, -1) < 1:
                continue
            if w[1] is None:
                act["take"] = p["id"]; return
            cur = w[1]["kind"]
            if (WEAPON_PREF.get(k, 0) > WEAPON_PREF.get(cur, 0) or cur == "mortar") and not (cur == "laser" and (w[1].get("charge") or 0) > 0):
                act["take"] = p["id"]; act["drop"] = 1; return


if __name__ == "__main__":
    mf.run(Bot)
