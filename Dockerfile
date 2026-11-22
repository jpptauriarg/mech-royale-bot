# Build from this folder: docker build -t my-bot .
# The match runs it with no network, a read-only root (writable /tmp), 1 CPU, 512 MB, and the map at /arena.
FROM python:3.13-slim
RUN pip install --no-cache-dir numpy==2.5.5 scipy==1.18.1
WORKDIR /bot
COPY mechfight.py world.py bot.py maps.npz ./
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
