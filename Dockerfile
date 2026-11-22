FROM python:3.13-slim
RUN pip install --no-cache-dir numpy==2.5.5 scipy==1.18.1
WORKDIR /bot
COPY mechfight.py world.py bot.py maps.npz ./
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
