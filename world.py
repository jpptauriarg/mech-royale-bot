"""Line of sight and ground height on the precomputed campus grids in maps.npz."""
import math

import numpy as np


class World:
    def __init__(self, path):
        m = np.load(path)
        self.x0, self.y0, nx, ny, cell = m["meta"]
        self.nx, self.ny = int(nx), int(ny)
        self.cell = float(cell)
        for key in ("G", "Bbase", "Btop", "Rbase", "Rtop", "Cbase", "Ctop", "Ttop", "SURF"):
            setattr(self, key, m[key])
        self.navblk = m["navblk"]
        self.navsurf = m["navsurf"]
        # Metres of crown that block sight. rules.canopyDensity 0.35, canopyBlockT 0.25.
        self.canopy_block = -math.log(0.25) / 0.35

    def idx(self, x, y):
        i = np.clip(np.rint(np.asarray(x) - self.x0).astype(np.int32), 0, self.nx - 1)
        j = np.clip(np.rint(np.asarray(y) - self.y0).astype(np.int32), 0, self.ny - 1)
        return i, j

    def surf(self, x, y):
        i, j = self.idx(x, y)
        return self.SURF[j, i]

    def los(self, a, b, canopy=True, step=2.0):
        """True where the segment from a to b is clear. a and b are (N, 3)."""
        a = np.atleast_2d(np.asarray(a, np.float64))
        b = np.atleast_2d(np.asarray(b, np.float64))
        delta = b - a
        dist = np.sqrt((delta * delta).sum(1))
        if dist.size == 0 or not np.isfinite(dist.max()) or dist.max() < 1e-3:
            return np.ones(len(a), dtype=bool)
        samples = int(max(2, math.ceil(float(dist.max()) / step)))
        t = (np.arange(1, samples) / samples)[None, :]
        px = a[:, 0:1] + delta[:, 0:1] * t
        py = a[:, 1:2] + delta[:, 1:2] * t
        pz = a[:, 2:3] + delta[:, 2:3] * t
        i, j = self.idx(px, py)
        blocked = pz < self.G[j, i] - 0.4
        blocked |= (pz >= self.Bbase[j, i]) & (pz <= self.Btop[j, i])
        blocked |= (pz >= self.Rbase[j, i]) & (pz <= self.Rtop[j, i])
        clear = ~blocked.any(1)
        if canopy:
            inside = (pz >= self.Cbase[j, i]) & (pz <= self.Ctop[j, i])
            trunk = pz <= self.Ttop[j, i]
            clear &= ~trunk.any(1)
            clear &= inside.sum(1) * (dist / samples) < self.canopy_block
        return clear

    def sees(self, eye, feet, canopy=True):
        """Eye (N, 3) can see a mech standing at feet (N, 3), centre or top."""
        feet = np.atleast_2d(np.asarray(feet, np.float64))
        mid = feet.copy()
        mid[:, 2] += 4.5
        top = feet.copy()
        top[:, 2] += 9.0
        return self.los(eye, mid, canopy=canopy) | self.los(eye, top, canopy=canopy)
