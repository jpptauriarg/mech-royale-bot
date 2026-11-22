# Match sandbox: no network, read-only root, writable /tmp, 1 CPU, 512 MB, map at /arena.
FROM python:3.13-slim
WORKDIR /bot
COPY mechfight.py bot.py nav.py holds.py ./
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
