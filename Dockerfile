FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml /app/
COPY README.md /app/
COPY agent /app/agent
COPY evals /app/evals
COPY tests /app/tests

# Install package and dependencies
RUN pip install --no-cache-dir -e .

# Expose port for reporting application dashboard
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default entrypoint: launch the reporting dashboard web server
CMD ["evals-report", "--host", "0.0.0.0", "--port", "8000"]
