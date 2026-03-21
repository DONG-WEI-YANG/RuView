FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN cd dashboard && npm ci && npx vite build
EXPOSE 8000 5005/udp
CMD ["python", "-m", "server", "--simulate"]
