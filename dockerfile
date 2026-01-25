# syntax=docker/dockerfile:1.7

########################
# 1️⃣ Builder stage
########################
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /wheels

# Install only what's needed to build wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Build wheels (cached)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-deps --wheel-dir /wheels -r requirements.txt


########################
# 2️⃣ Runtime stage
########################
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-root user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install wheels only (no internet, no build)
RUN pip install \
    --no-cache-dir \
    --no-index \
    --find-links=/wheels \
    /wheels/* \
    && rm -rf /wheels

# Copy app last (best cache usage)
COPY . .

USER appuser

EXPOSE 8000

CMD ["gunicorn", "app:app", \
     "--bind=0.0.0.0:8000", \
     "--workers=2", \
     "--threads=2", \
     "--timeout=30"]

ENV PYTHONHASHSEED=0
