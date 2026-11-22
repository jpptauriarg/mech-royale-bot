import sys, math, time, numpy as np
sys.path.insert(0, "repo")
from world import World
W = World("repo/maps.npz")
NX4, NY4 = W.nx // 4, W.ny // 4
xs = W.x0 + 2 + 4 * np.arange(NX4); ys = W.y0 + 2 + 4 * np.arange(NY4)
X, Y = np.meshgrid(xs, ys)
nb = W.navblk[:NY4 * 2:2, :NX4 * 2:2]
cells = np.argwhere(~nb)
expo = np.full((NY4, NX4), 1.0, np.float32)
dists = np.array([25, 45, 70, 100, 140, 185]); weights = np.array([1.5, 1.4, 1.2, 1.0, 0.8, 0.6])
dirs = np.arange(16) * (2 * math.pi / 16)
OX = (np.cos(dirs)[:, None] * dists[None, :]).ravel(); OY = (np.sin(dirs)[:, None] * dists[None, :]).ravel()
WT = np.tile(weights, 16); NEAR = np.tile(dists <= 45, 16)
t0 = time.time()
CH = 400
for s in range(0, len(cells), CH):
    cc = cells[s:s + CH]
    cx = X[cc[:, 0], cc[:, 1]]; cy = Y[cc[:, 0], cc[:, 1]]
    tx = np.repeat(cx, len(OX)); ty = np.repeat(cy, len(OX))
    ox = tx + np.tile(OX, len(cc)); oy = ty + np.tile(OY, len(cc))
    oi, oj = W.idx(ox, oy)
    valid = ~W.navblk[np.clip(oj // 2, 0, W.navblk.shape[0] - 1), np.clip(oi // 2, 0, W.navblk.shape[1] - 1)]
    eye = np.stack([ox, oy, W.surf(ox, oy) + 8], 1)
    feet = np.stack([tx, ty, W.surf(tx, ty)], 1)
    near = np.tile(NEAR, len(cc))
    vis = np.zeros(len(tx), bool)
    vis[near] = W.sees(eye[near], feet[near], canopy=False)
    vis[~near] = W.sees(eye[~near], feet[~near], canopy=True)
    w = np.tile(WT, len(cc)) * valid
    v = (vis * w).reshape(len(cc), -1).sum(1); tot = w.reshape(len(cc), -1).sum(1)
    expo[cc[:, 0], cc[:, 1]] = np.where(tot > 0, v / np.maximum(tot, 1e-6), 1.0)
    if s % 8000 == 0: print(s, len(cells), time.time() - t0, flush=True)
m = dict(np.load("repo/maps.npz")); m["expo"] = expo
np.savez_compressed("repo/maps.npz", **m)
print("done", time.time() - t0, expo[~nb].mean())
