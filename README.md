# shade

Survival-first bot for the Builders Buildathon Mech Royale track (`mechfight/1`). It plays for placement: rotate into the next circle early, keep buildings between itself and other mechs, walk quietly when someone is close, dodge incoming fire, and shoot only when a fight is already forced.

## License

MIT. See `LICENSE`.

## Libraries

Python 3 standard library only. No third-party packages.

`mechfight.py` is the official kit wrapper (standard library, stdin/stdout loop and debug drawing), copied in as the kit allows. `check/check.py`, `mechfight.schema.json`, and `arena/` are from that same kit and are used to conformance-check the bot locally. The bot itself (`bot.py`, `nav.py`) and the local placement harness (`harness/sim.py`) are this project's code.

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
