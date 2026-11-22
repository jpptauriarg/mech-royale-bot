# Mech Royale bot: team "Just for the fun of it"

A mechfight/1 bot for the Builders Buildathon Mech Royale track.

## What it does
- Loads `maps.npz`, grids precomputed from the kit's map (`arena/v2`) by `tools/build_maps.py` and `tools/build_expo.py`: walk surface, building/bridge/tree blockers, a 2 m navigation grid, and a per cell "exposure" score (how visible a spot is from its surroundings).
- Picks low exposure hiding cells inside the next zone circle and walks there by Dijkstra on the nav grid (paths prefer covered cells).
- Fights hunters one at a time: leads gun shots, charges and releases the laser on target, zigzag strafes in a range band, retreats out of line of sight to regenerate when hurt or outnumbered, dodges incoming bullets and aimed lasers.
- Picks up laser or lightning, collects nearby cubes, picks durability upgrades first (plating, armor, nanites).
- Jumps and jets out of map pockets (stadium bowl, courtyards) and when stuck.

## Credits and libraries
- `mechfight.py` is the protocol wrapper from the official mechfight bot kit (Builders), used unchanged.
- The map data in `maps.npz` is derived from the kit's `arena/v2` files.
- [NumPy](https://numpy.org) (BSD-3-Clause) and [SciPy](https://scipy.org) (BSD-3-Clause) for grids, shortest paths and connected components.
- Everything else is our own code, written with AI assistance.

## Run
```sh
docker build -t my-bot .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" my-bot
```
