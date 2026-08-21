#!/usr/bin/env bash
# Start the web app and the compute service together, bound so that a phone on
# the same wifi can reach them.
#
#   ./serve.sh
#
# Ctrl-C stops both. See docs/running_the_web_app.md for the Windows-side step
# that WSL2 needs before the LAN can actually reach these ports.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

WEB_PORT=${WEB_PORT:-3111}
API_PORT=${API_PORT:-8000}

# The pipeline's interpreter, not whatever is first on PATH — torch, VGGT and
# GroundingDINO live in this one environment.
PYTHON=${PYTHON:-"$HOME/miniconda3/envs/senior/bin/python"}
if [ ! -x "$PYTHON" ]; then
  echo "ERROR: no interpreter at $PYTHON — set PYTHON=/path/to/python" >&2
  exit 1
fi

if [ ! -d web/.next ]; then
  echo "Building the web app (first run only)…"
  (cd web && npm run build)
fi

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

"$PYTHON" -m uvicorn service.app:app --host 0.0.0.0 --port "$API_PORT" &
(cd web && npx next start -H 0.0.0.0 -p "$WEB_PORT") &

WSL_IP=$(hostname -I | awk '{print $1}')
cat <<EOF

  web   http://localhost:$WEB_PORT
  api   http://localhost:$API_PORT

  From another device on the wifi, open the web port on the Windows host's LAN
  address. WSL2 is behind its own NAT, so that only works after forwarding the
  ports once, from an ADMINISTRATOR PowerShell on Windows:

    netsh interface portproxy add v4tov4 listenport=$WEB_PORT listenaddress=0.0.0.0 connectport=$WEB_PORT connectaddress=$WSL_IP
    netsh interface portproxy add v4tov4 listenport=$API_PORT listenaddress=0.0.0.0 connectport=$API_PORT connectaddress=$WSL_IP
    New-NetFirewallRule -DisplayName "cubit" -Direction Inbound -Protocol TCP -LocalPort $WEB_PORT,$API_PORT -Action Allow

  This WSL address ($WSL_IP) changes on reboot; docs/running_the_web_app.md has
  the durable alternative.

EOF

wait
