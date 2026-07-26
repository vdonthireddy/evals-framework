#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🚀 Starting Evals Framework & Model Reporting Application"
echo "============================================================"

# Ensure .env file exists
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "--> Creating .env from .env.example..."
        cp .env.example .env
    else
        echo "--> Creating default .env file..."
        touch .env
    fi
fi

# Ensure storage directory exists and seed initial evaluation runs if empty
mkdir -p evals/results
python3 -m evals.store.seed_store evals/results/eval_results.db 2>/dev/null || true

# Determine launch method (Docker vs Local Python Fallback)
if command -v docker &> /dev/null && docker info &> /dev/null; then
    echo "--> Launching via Docker Container..."
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d --build
    elif docker compose version &> /dev/null; then
        docker compose up -d --build
    else
        echo "--> Building Docker image evals-framework..."
        docker build -t evals-framework .
        docker stop evals-framework-app 2>/dev/null || true
        docker rm evals-framework-app 2>/dev/null || true
        docker run -d --name evals-framework-app -p 8000:8000 -v "$PWD/evals/results:/app/evals/results" --env-file .env evals-framework
    fi
    echo "✅ Container successfully launched!"
else
    echo "--> Docker daemon not detected. Launching via local Python web server..."
    pkill -f "evals.cli report" 2>/dev/null || true
    python3 -m evals.cli report --port 8000 > /dev/null 2>&1 &
    echo "✅ Local server launched in background (PID $!)"
fi

echo ""
echo "============================================================"
echo "📊 Evals Reporting Dashboard Ready!"
echo "👉 Open UI at: http://localhost:8000"
echo "============================================================"
