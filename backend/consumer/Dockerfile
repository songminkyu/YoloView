FROM python:3.9-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN uv pip install --system .

COPY . .

CMD ["python", "worker.py"]
