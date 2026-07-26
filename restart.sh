#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🔄 Restarting Evals Framework & Reporting Application"
echo "============================================================"

# Stop running services
./stop.sh

# Short pause to allow ports to free up
sleep 1

# Start services
./start.sh
