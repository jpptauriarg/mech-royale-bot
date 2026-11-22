# Oval Ghost

Survival bot for the Builders Buildathon Mech Royale track (protocol `mechfight/1`).

Practice matches are 15 house bots that hunt only this mech, so the score is how many of them die before we do. The bot paths on a navigation grid to the nearest hidden cell inside the next safe circle. A gunshot or a boost it can hear starts that run before the hunter is in view. It breaks line of sight and regenerates when more than one hunter can see it, and it duels a single close mech from a strafe. Incoming bullets only nudge the path when they are about to hit. It picks up a laser or lightning, and takes durability upgrades in this order: hit points, armor, regen.

`maps.npz` is built from the official arena by `tools/build_maps.py` (walk surface, building and bridge slabs, canopy, and a 4 m navigation grid). The match still mounts the live map at `/arena`; this bot uses the precomputed grids so a tick stays well under 50 ms.

## Run

```sh
python3 tools/build_maps.py /path/to/arena maps.npz
docker build -t oval-ghost .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" oval-ghost
```

## License

MIT. See `LICENSE`.

## Libraries

- Python 3 standard library.
- `mechfight.py` is the official mechfight starter-kit wrapper (stdin/stdout loop and message types), included unchanged.
- [NumPy](https://numpy.org) (BSD-3-Clause) and [SciPy](https://scipy.org) (BSD-3-Clause) for the map grids, shortest paths, and connected components.
