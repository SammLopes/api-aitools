# backend/Dockerfile
FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    vim\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY main.py .
COPY swagger_config.py .
COPY docs ./docs
COPY model ./model

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3000
ENV FLASK_APP=main.py

CMD ["flask", "run", "--host=0.0.0.0", "--port=3000", "--debug"]
