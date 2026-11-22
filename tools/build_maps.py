"""Offline: rasterize world.json + terrain into grids the bot loads fast (maps.npz)."""
import json, sys, math
import numpy as np

arena = sys.argv[1]; out = sys.argv[2]
w = json.load(open(f"{arena}/v2/world.json"))
T = w["terrain"]; nx, ny = T["nx"], T["ny"]; x0, y0 = T["minX"], T["minY"]
G = np.fromfile(f"{arena}/v2/terrain.i16", dtype="<i2").astype(np.float32).reshape(ny, nx) * 0.01

def poly_mask(flat, xs, ys):
    px = np.array(flat[0::2]); py = np.array(flat[1::2])
    X, Y = np.meshgrid(xs, ys)
    inside = np.zeros(X.shape, bool)
    n = len(px)
    for k in range(n):
        xa, ya, xb, yb = px[k], py[k], px[(k + 1) % n], py[(k + 1) % n]
        cond = (ya > Y) != (yb > Y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = xa + (Y - ya) * (xb - xa) / (yb - ya)
        inside ^= cond & (X < xint)
    return inside

NEG = -1e4
Bbase = np.full((ny, nx), 1e4, np.float32); Btop = np.full((ny, nx), NEG, np.float32)
Rbase = np.full((ny, nx), 1e4, np.float32); Rtop = np.full((ny, nx), NEG, np.float32)
WALL = np.zeros((ny, nx), bool)        # blocks walking at ground level
SURF = G.copy()                        # walk surface height
for p in w["prisms"]:
    r = p["ring"]; px = r[0::2]; py = r[1::2]
    i0 = max(0, int(math.floor(min(px) - x0)) - 1); i1 = min(nx, int(math.ceil(max(px) - x0)) + 2)
    j0 = max(0, int(math.floor(min(py) - y0)) - 1); j1 = min(ny, int(math.ceil(max(py) - y0)) + 2)
    if i1 <= i0 or j1 <= j0: continue
    xs = x0 + np.arange(i0, i1); ys = y0 + np.arange(j0, j1)
    m = poly_mask(r, xs, ys)
    for h in p["holes"]:
        m &= ~poly_mask(h, xs, ys)
    if not m.any(): continue
    base, top = p["base"], p["top"]
    g = G[j0:j1, i0:i1]
    if p["kind"] == "bridge":
        sub = Rbase[j0:j1, i0:i1]; sub[m] = np.minimum(sub[m], base)
        sub = Rtop[j0:j1, i0:i1]; sub[m] = np.maximum(sub[m], top)
    else:
        sub = Bbase[j0:j1, i0:i1]; sub[m] = np.minimum(sub[m], base)
        sub = Btop[j0:j1, i0:i1]; sub[m] = np.maximum(sub[m], top)
    s = SURF[j0:j1, i0:i1]; wl = WALL[j0:j1, i0:i1]
    low = m & (top <= g + 1.2) & (top > s)          # platform you step onto
    s[low] = top
    wallc = m & (top > g + 1.2) & (base < g + 9.5)
    wl[wallc] = True

# trees
f = w["trees"]["fields"]; d = np.array(w["trees"]["data"], np.float64).reshape(-1, len(f)); F = {k: i for i, k in enumerate(f)}
Cbase = np.full((ny, nx), 1e4, np.float32); Ctop = np.full((ny, nx), NEG, np.float32)
Ttop = np.full((ny, nx), NEG, np.float32)
for t in d:
    z = t[F["z"]]; hgt = t[F["height"]]; cr = t[F["crownR"]]; cb = t[F["crownBase"]]; tr = t[F["trunkR"]]
    cx, cy = t[F["crownX"]], t[F["crownY"]]
    i0 = max(0, int(cx - cr - x0) - 1); i1 = min(nx, int(cx + cr - x0) + 2)
    j0 = max(0, int(cy - cr - y0) - 1); j1 = min(ny, int(cy + cr - y0) + 2)
    X, Y = np.meshgrid(x0 + np.arange(i0, i1), y0 + np.arange(j0, j1))
    m = (X - cx) ** 2 + (Y - cy) ** 2 <= cr * cr
    sb = Cbase[j0:j1, i0:i1]; sb[m] = np.minimum(sb[m], z + cb)
    st = Ctop[j0:j1, i0:i1]; st[m] = np.maximum(st[m], z + hgt)
    tx, ty = t[F["x"]], t[F["y"]]
    i = int(round(tx - x0)); j = int(round(ty - y0))
    if 0 <= i < nx and 0 <= j < ny:
        Ttop[j, i] = max(Ttop[j, i], z + max(cb, hgt * 0.6))

# slope blocking on the walk surface
gy, gx = np.gradient(SURF)
steep = np.hypot(gx, gy) > math.tan(math.radians(50))
steep &= ~WALL
# map margin
edge = np.zeros_like(WALL); edge[:3, :] = edge[-3:, :] = True; edge[:, :3] = edge[:, -3:] = True
B = w["bounds"]
X, Y = np.meshgrid(x0 + np.arange(nx), y0 + np.arange(ny))
edge |= (X < B["minX"] + 3) | (X > B["maxX"] - 3) | (Y < B["minY"] + 3) | (Y > B["maxY"] - 3)
blocked = WALL | steep | edge
# dilate by mech radius ~3 m (disk)
from scipy.ndimage import binary_dilation
yy, xx = np.mgrid[-3:4, -3:4]; disk = xx ** 2 + yy ** 2 <= 9.5
blk = binary_dilation(WALL | edge, structure=disk) | binary_dilation(steep, structure=np.ones((3, 3), bool))
# 2 m nav grid: blocked if any 1 m cell blocked
NX2, NY2 = nx // 2, ny // 2
nb = blk[:NY2 * 2, :NX2 * 2].reshape(NY2, 2, NX2, 2).any(axis=(1, 3))
ns = SURF[:NY2 * 2, :NX2 * 2].reshape(NY2, 2, NX2, 2).mean(axis=(1, 3))
np.savez_compressed(out, G=G.astype(np.float16) if False else G, Bbase=Bbase, Btop=Btop, Rbase=Rbase, Rtop=Rtop,
                    Cbase=Cbase, Ctop=Ctop, Ttop=Ttop, SURF=SURF, navblk=nb, navsurf=ns.astype(np.float32),
                    meta=np.array([x0, y0, nx, ny], np.float64))
print("wall frac", WALL.mean(), "steep frac", steep.mean(), "nav blocked", nb.mean())
