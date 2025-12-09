#!/usr/bin/env zsh

# Launch TensorBoard to view training progress
# Usage: ./launch_tensorboard.sh [optional_port]

PORT=${1:-6006}

LATEST_DIR=$(ls -d results/*_results 2>/dev/null | sort -r | head -1)

if [[ -z "$LATEST_DIR" ]]; then
    echo "❌ No results directory found in results/"
    exit 1
fi

echo "📁 Found latest results: $LATEST_DIR"

LOGDIR="$LATEST_DIR/tensorboard_logs"

if [[ ! -d "$LOGDIR" ]]; then
    echo "❌ TensorBoard logs not found at: $LOGDIR"
    exit 1
fi

echo "🚀 Launching TensorBoard..."
echo "   Log directory: $LOGDIR"
echo "   Port: $PORT"
echo ""
echo "Open in browser: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""

tensorboard --logdir=$LOGDIR --port=$PORT --bind_all
