FROM python:3.12-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch to keep image small)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY server/ server/
COPY dashboard/ dashboard/
COPY models/ models/
COPY pyproject.toml .

EXPOSE 8000
EXPOSE 5005/udp

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "server"]
