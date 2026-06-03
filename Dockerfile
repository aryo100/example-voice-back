FROM python:3.10-slim

WORKDIR /app

# ffmpeg for optional MP3 recording; gcc for webrtcvad build.
# No database server in this image — PostgreSQL and object storage are external.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY assistant ./assistant
COPY run.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
