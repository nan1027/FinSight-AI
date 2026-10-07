FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.12.0 \
    && pip install --no-cache-dir xgboost==3.0.5 --no-deps \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY backend ./backend
COPY rag ./rag
COPY ml ./ml
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV PORT=8080

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]