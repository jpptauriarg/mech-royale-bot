# Oval Ghost

Cover-first survival bot for the Builders Buildathon Mech Royale track (protocol `mechfight/1`).

Placement is the score, and the practice house bots hunt us. The bot plays for that:

- **Reads the map.** At startup it builds a 2 m height grid from `/arena` (`arena.py`, about 0.15 s) and uses it for line of sight, line of fire, and walking paths.
- **Survives the zone.** It plans an A* path into the next safe circle and waits there in a well-enclosed spot (walls on most sides), chosen away from enemies it knows about. Paths avoid known enemies too.
- **Uses cover.** Every few ticks it scores nearby spots by who could see it there. It hides from every known threat unless it is ready to fight; caught in the open, it runs for the nearest hidden spot, or for its enclosed camp.
- **Fights from cover, one target at a time.** Loaded and in range, it shoots its target (whoever is hitting it first) from the best spot it can find, and never walks away from a target that is nearly dead (hiding only lets it regenerate). Reloading, cooling down, or hurt with nothing in range, it ducks back out of sight and regenerates.
- **Gets a laser.** It remembers every laser it has seen, explores for one early, and loots dead enemies. In the simulator a laser is the difference between ~0.4 and ~4.4 points per match. Lightning is second choice; it never picks up mortars (a laser hit on a mortar carrier cooks it off).
- **Shoots well.** It leads targets with the gun, aims a laser 0.1 s ahead and releases it only on target, faces attackers it cannot see, and fires at whatever part of a target is out of cover.
- **Stays fast.** Heavy work (cover search, path planning) is capped per tick and spread over ticks. It answers in about 1 ms typically and under 25 ms worst case in the 1-CPU sandbox.

## Run

The image entrypoint is `python3 -u /bot/bot.py`. The match mounts the map at `/arena` and speaks `mechfight/1` on stdin and stdout. Without the map the bot still plays, just without cover.

Local conformance check, from the official starter kit:

```sh
python3 check/check.py -- python3 -u /path/to/bot.py
docker build -t oval-ghost .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" oval-ghost
```

## Local simulator

`sim/mechsim.py` is a rough stand-in for a practice match, for comparing bot versions. It pits the bot against 15 hunters that grab lasers and close in. It is not the real server: read its numbers as relative, not absolute.

```sh
python3 sim/mechsim.py --kit /path/to/mechfight-kit --matches 12 -- python3 -u bot.py
```

## License

MIT. See `LICENSE`.

## Libraries

Python 3 standard library only. `mechfight.py` is the official mechfight starter-kit wrapper (stdin/stdout loop and message types), included unchanged. No other third-party libraries.
