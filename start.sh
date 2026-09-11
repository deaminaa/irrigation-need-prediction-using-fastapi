#!/bin/bash
#
# Runs FastAPI and Streamlit side by side as PID 1's children.
#
# Neither process may outlive the other. A Streamlit UI whose API has died
# renders a connection error on every prediction while the container still
# reports itself up, so Docker's restart policy never fires. Exiting on the
# first death is what makes that policy work.

set -uo pipefail

api_pid=""
ui_pid=""

shutdown() {
    [[ -n "$api_pid" ]] && kill -TERM "$api_pid" 2>/dev/null
    [[ -n "$ui_pid" ]] && kill -TERM "$ui_pid" 2>/dev/null
    return 0
}

trap shutdown TERM INT

echo "Starting FastAPI..."
uvicorn main:app --host 127.0.0.1 --port 8000 &
api_pid=$!

echo "Starting Streamlit..."
streamlit run irrigation.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false &
ui_pid=$!

wait -n
code=$?

echo "A service exited with status ${code}; shutting down the container."
shutdown
wait
exit "$code"
