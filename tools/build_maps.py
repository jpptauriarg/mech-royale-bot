"""Rasterize the kit arena into maps.npz. Run from the repo root:

    python3 tools/build_maps.py /path/to/arena maps.npz
"""
import math
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation

arena, out = sys.argv[1], sys.argv[2]
t0 = time.perf_counter()
import json

with open(f"{arena}/v2/world.json") as f:
    world = json.load(f)
terrain = world["terrain"]
nx, ny = terrain["nx"], terrain["ny"]
x0, y0 = float(terrain["minX"]), float(terrain["minY"])
ground = np.fromfile(f"{arena}/v2/terrain.i16", dtype="<i2").astype(np.float32).reshape(ny, nx) * 0.01

NEG = np.float32(-1e4)
POS = np.float32(1e4)


def poly_mask(flat, xs, ys):
    px = np.asarray(flat[0::2], np.float64)
    py = np.asarray(flat[1::2], np.float64)
    x, y = np.meshgrid(xs, ys)
    inside = np.zeros(x.shape, dtype=bool)
    n = len(px)
    for k in range(n):
        x1, y1 = px[k], py[k]
        x2, y2 = px[(k + 1) % n], py[(k + 1) % n]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (x < xint)
    return inside


bbase = np.full((ny, nx), POS, np.float32)
btop = np.full((ny, nx), NEG, np.float32)
rbase = np.full((ny, nx), POS, np.float32)
rtop = np.full((ny, nx), NEG, np.float32)
wall = np.zeros((ny, nx), dtype=bool)
surf = ground.copy()

for prism in world["prisms"]:
    ring = prism["ring"]
    xs_w = ring[0::2]
    ys_w = ring[1::2]
    i0 = max(0, int(math.floor(min(xs_w) - x0)) - 1)
    i1 = min(nx, int(math.ceil(max(xs_w) - x0)) + 2)
    j0 = max(0, int(math.floor(min(ys_w) - y0)) - 1)
    j1 = min(ny, int(math.ceil(max(ys_w) - y0)) + 2)
    if i1 <= i0 or j1 <= j0:
        continue
    xs = x0 + np.arange(i0, i1)
    ys = y0 + np.arange(j0, j1)
    mask = poly_mask(ring, xs, ys)
    for hole in prism.get("holes") or []:
        mask &= ~poly_mask(hole, xs, ys)
    if not mask.any():
        continue
    base, top = float(prism["base"]), float(prism["top"])
    g = ground[j0:j1, i0:i1]
    if prism.get("kind") == "bridge":
        rbase[j0:j1, i0:i1][mask] = np.minimum(rbase[j0:j1, i0:i1][mask], base)
        rtop[j0:j1, i0:i1][mask] = np.maximum(rtop[j0:j1, i0:i1][mask], top)
        continue
    bbase[j0:j1, i0:i1][mask] = np.minimum(bbase[j0:j1, i0:i1][mask], base)
    btop[j0:j1, i0:i1][mask] = np.maximum(btop[j0:j1, i0:i1][mask], top)
    low = mask & (top <= g + 1.2) & (top > surf[j0:j1, i0:i1])
    surf[j0:j1, i0:i1][low] = top
    wall[j0:j1, i0:i1] |= mask & (top > g + 1.2) & (base < g + 9.5)

cbase = np.full((ny, nx), POS, np.float32)
ctop = np.full((ny, nx), NEG, np.float32)
ttop = np.full((ny, nx), NEG, np.float32)
fields = world["trees"]["fields"]
data = np.asarray(world["trees"]["data"], np.float64).reshape(-1, len(fields))
col = {name: i for i, name in enumerate(fields)}
for tree in data:
    cr = tree[col["crownR"]]
    cx, cy = tree[col["crownX"]], tree[col["crownY"]]
    i0 = max(0, int(cx - cr - x0) - 1)
    i1 = min(nx, int(cx + cr - x0) + 2)
    j0 = max(0, int(cy - cr - y0) - 1)
    j1 = min(ny, int(cy + cr - y0) + 2)
    if i1 > i0 and j1 > j0:
        gx, gy = np.meshgrid(x0 + np.arange(i0, i1), y0 + np.arange(j0, j1))
        mask = (gx - cx) ** 2 + (gy - cy) ** 2 <= cr * cr
        z = tree[col["z"]]
        cbase[j0:j1, i0:i1][mask] = np.minimum(cbase[j0:j1, i0:i1][mask], z + tree[col["crownBase"]])
        ctop[j0:j1, i0:i1][mask] = np.maximum(ctop[j0:j1, i0:i1][mask], z + tree[col["height"]])
    ti = int(round(tree[col["x"]] - x0))
    tj = int(round(tree[col["y"]] - y0))
    if 0 <= ti < nx and 0 <= tj < ny:
        ttop[tj, ti] = max(ttop[tj, ti], tree[col["z"]] + max(tree[col["crownBase"]], tree[col["height"]] * 0.6))

gy, gx = np.gradient(surf)
steep = np.hypot(gx, gy) > math.tan(math.radians(50))
bounds = world["bounds"]
xx, yy = np.meshgrid(x0 + np.arange(nx), y0 + np.arange(ny))
edge = (
    (xx < bounds["minX"] + 4)
    | (xx > bounds["maxX"] - 4)
    | (yy < bounds["minY"] + 4)
    | (yy > bounds["maxY"] - 4)
)
disk_y, disk_x = np.mgrid[-3:4, -3:4]
disk = disk_x ** 2 + disk_y ** 2 <= 10
blocked = binary_dilation(wall | edge, structure=disk) | (steep & ~wall)

cell = 4
nh, nw = ny // cell, nx // cell
navblk = blocked[: nh * cell, : nw * cell].reshape(nh, cell, nw, cell).any(axis=(1, 3))
navsurf = surf[: nh * cell, : nw * cell].reshape(nh, cell, nw, cell).mean(axis=(1, 3)).astype(np.float32)

np.savez_compressed(
    out,
    G=ground,
    Bbase=bbase,
    Btop=btop,
    Rbase=rbase,
    Rtop=rtop,
    Cbase=cbase,
    Ctop=ctop,
    Ttop=ttop,
    SURF=surf,
    navblk=navblk,
    navsurf=navsurf,
    meta=np.array([x0, y0, nx, ny, cell], np.float64),
)
print(
    f"wrote {out} wall {wall.mean():.3f} steep {steep.mean():.3f} nav {navblk.mean():.3f} "
    f"in {time.perf_counter() - t0:.1f}s",
    flush=True,
)
