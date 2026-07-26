#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🛑 Stopping Evals Framework & Reporting Application"
echo "============================================================"

# Stop Docker containers if active
if command -v docker &> /dev/null && docker info &> /dev/null; then
    if command -v docker-compose &> /dev/null; then
        docker-compose down 2>/dev/null || true
    elif docker compose version &> /dev/null; then
        docker compose down 2>/dev/null || true
    else
        docker stop evals-framework-app 2>/dev/null || true
        docker rm evals-framework-app 2>/dev/null || true
    fi
fi

# Stop local background server if running
pkill -f "evals.cli report" 2>/dev/null || true
pkill -f "evals-report" 2>/dev/null || true

echo "✅ All services stopped successfully."
