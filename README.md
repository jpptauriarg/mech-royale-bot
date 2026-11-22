# Oval Ghost

Survival bot for the Builders Buildathon Mech Royale track (protocol `mechfight/1`).

Placement is the score. The mech walks into the next safe circle early enough to arrive on foot, and it boosts only when the ring would catch it or a mech is on top of it. It keeps its distance from other mechs and NPC drones, stays under step-sound speed while it is already safe, sidesteps incoming projectiles, and fires only at a lone close threat or in the final circles.

## Run

The image entrypoint is `python3 -u /bot/bot.py`. The match mounts the map at `/arena` and speaks `mechfight/1` on stdin and stdout. This bot does not read the map files.

Local conformance check, from a checkout of the official starter kit:

```sh
python3 check/check.py -- python3 -u /path/to/bot.py
docker build -t oval-ghost .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" oval-ghost
```

## License

MIT. See `LICENSE`.

## Libraries

Python 3 standard library only. `mechfight.py` is the official mechfight starter-kit wrapper (stdin/stdout loop and message types), included unchanged. No other third-party libraries.
