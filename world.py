"""Map model: grids built from /arena by tools/build_maps.py, line-of-sight and ground queries."""
import math
import numpy as np


class World:
    def __init__(self, path):
        m = np.load(path)
        self.x0, self.y0, nx, ny = m["meta"]
        self.nx, self.ny = int(nx), int(ny)
        for k in ("G", "Bbase", "Btop", "Rbase", "Rtop", "Cbase", "Ctop", "Ttop", "SURF"):
            setattr(self, k, m[k])
        self.navblk = m["navblk"]
        self.navsurf = m["navsurf"]
        self.expo = m["expo"] if "expo" in m.files else None
        self.canopy_block = -math.log(0.25) / 0.35  # metres of crown that stop sight (guess from rules)

    def idx(self, x, y):
        i = np.clip(np.rint(np.asarray(x) - self.x0).astype(np.int64), 0, self.nx - 1)
        j = np.clip(np.rint(np.asarray(y) - self.y0).astype(np.int64), 0, self.ny - 1)
        return i, j

    def surf(self, x, y):
        i, j = self.idx(x, y)
        return self.SURF[j, i]

    def los(self, a, b, canopy=True, step=1.5):
        """a, b: (N,3) arrays. True where the segment is clear of terrain/buildings (and dense canopy)."""
        a = np.atleast_2d(np.asarray(a, np.float64)); b = np.atleast_2d(np.asarray(b, np.float64))
        d = b - a
        L = np.sqrt((d ** 2).sum(1))
        S = int(max(2, math.ceil(L.max() / step)))
        t = (np.arange(1, S) / S)[None, :]
        px = a[:, 0:1] + d[:, 0:1] * t; py = a[:, 1:2] + d[:, 1:2] * t; pz = a[:, 2:3] + d[:, 2:3] * t
        i, j = self.idx(px, py)
        blocked = (pz < self.G[j, i] - 0.3)
        blocked |= (pz >= self.Bbase[j, i]) & (pz <= self.Btop[j, i])
        blocked |= (pz >= self.Rbase[j, i]) & (pz <= self.Rtop[j, i])
        ok = ~blocked.any(1)
        if canopy:
            seg = (L / S)
            inside = (pz >= self.Cbase[j, i]) & (pz <= self.Ctop[j, i])
            trunk = pz <= self.Ttop[j, i]
            ok &= ~trunk.any(1)
            ok &= inside.sum(1) * seg < self.canopy_block
        return ok

    def sees(self, eye, tgt_feet, h=9.0, canopy=True):
        """Observer eye(s) (N,3) sees mech(s) with feet tgt_feet (N,3): centre or top."""
        tf = np.atleast_2d(np.asarray(tgt_feet, np.float64))
        c = tf.copy(); c[:, 2] += h * 0.5
        top = tf.copy(); top[:, 2] += h
        return self.los(eye, c, canopy) | self.los(eye, top, canopy)
