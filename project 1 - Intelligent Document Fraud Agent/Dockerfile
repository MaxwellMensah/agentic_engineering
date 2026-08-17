# STAGE 1: BUILDER
FROM python:3.11-slim AS builder

WORKDIR /app

# Prevent Python from writing .pyc files and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies into a temporary local folder
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# STAGE 2: PRODUCTION RUNNER
FROM python:3.11-slim AS runner

WORKDIR /app

# 1. Create a non-root user with a home directory
RUN adduser --disabled-password --gecos "" appuser

# 2. Copy installed packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# 3. Copy application code and set ownership
COPY --chown=appuser:appuser . .

# 4. Set environment variables so python finds installed packages
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 5. Switch away from root user
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]