# shade

Survival-first bot for the Builders Buildathon Mech Royale track (`mechfight/1`). It plays for placement. The practice squad hunts only this mech and stays on the ground, so the bot jets onto a roof they cannot path onto and stays there until the zone forces a move. On the way it rotates early, dodges, and shoots only when a fight is already forced.

Enclosed courtyards were checked and are not real pockets on this map: almost every building hole is filled by another prism. A roof blocks a ground eye standing at the wall. An eye far enough that the line clears the roof edge still sees the mech, because vision ignores pitch.

## License

MIT. See `LICENSE`.

## Libraries

Python 3 standard library only. No third-party packages.

`mechfight.py` is the official kit wrapper (standard library, stdin/stdout loop and debug drawing), copied in as the kit allows. `check/check.py`, `mechfight.schema.json`, and `arena/` are from that same kit and are used to conformance-check the bot locally. The bot itself (`bot.py`, `nav.py`, `holds.py`) and the local placement harness (`harness/sim.py`) are this project's code.

## Run

```sh
python3 check/check.py -- python3 -u bot.py
docker build -t shade .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" shade
```

Local placement harness (not the official scorer; gunfights, zone, and buildings only):

```sh
python3 harness/sim.py --mode ffa --matches 12
python3 harness/sim.py --mode squad --matches 12
```
