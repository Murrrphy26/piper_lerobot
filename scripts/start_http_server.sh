#!/usr/bin/env bash
# 启动无显示器 HTTP 控制服务，并为本次运行创建独立日志目录。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

export HTTP_SERVER_LOG_DIR="${REPO_ROOT}/logs/http_server/${RUN_TIMESTAMP}"
HTTP_SERVER_HOST="${HTTP_SERVER_HOST:-0.0.0.0}"
HTTP_SERVER_PORT="${HTTP_SERVER_PORT:-12123}"

mkdir -p "${HTTP_SERVER_LOG_DIR}"

cd "${REPO_ROOT}"
echo "Starting Piper HTTP Control Server."
echo "  address: http://${HTTP_SERVER_HOST}:${HTTP_SERVER_PORT}"
echo "  logs: ${HTTP_SERVER_LOG_DIR}"

exec python -m uvicorn http_server.app:create_app \
  --factory \
  --host "${HTTP_SERVER_HOST}" \
  --port "${HTTP_SERVER_PORT}" \
  --access-log \
  >>"${HTTP_SERVER_LOG_DIR}/http.log" 2>&1
