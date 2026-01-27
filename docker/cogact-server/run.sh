#!/bin/bash
# CogACT server helper script (wraps docker-compose)
#
# Usage:
#   ./docker/cogact-server/run.sh         # Start server
#   ./docker/cogact-server/run.sh stop    # Stop server
#   ./docker/cogact-server/run.sh logs    # View logs
#   ./docker/cogact-server/run.sh build   # Rebuild image

set -e
cd "$(dirname "$0")/../.."

case "${1:-up}" in
    up|start)
        echo "Starting CogACT server (GPU 0, port 5500)..."
        docker compose up -d cogact
        echo ""
        echo "Server starting... (model load takes ~2 min)"
        echo "  Logs:   docker compose logs -f cogact"
        echo "  Health: curl http://localhost:5500/health"
        echo "  Stop:   docker compose down"
        echo ""
        echo "Run simulation with:"
        echo "  export COGACT_SERVER_URL=http://127.0.0.1:5500/act_batch"
        echo "  CUDA_VISIBLE_DEVICES=1 xvfb-run -a python examples/warehouse_scenario.py --vla cogact_server"
        ;;
    stop|down)
        docker compose down
        ;;
    logs)
        docker compose logs -f cogact
        ;;
    build)
        docker compose build cogact
        ;;
    restart)
        docker compose restart cogact
        ;;
    *)
        echo "Usage: $0 [up|stop|logs|build|restart]"
        exit 1
        ;;
esac
