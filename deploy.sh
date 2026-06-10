#!/usr/bin/env bash
# Pull latest code and (re)deploy the Traefik-fronted container.
# Run this from the repo root on the server. Uses only docker-compose.yml so the
# local-dev docker-compose.override.yml (bind-mount + published port) is ignored.
set -euo pipefail

cd "$(dirname "$0")"

echo "==> git pull"
git pull

echo "==> git lfs pull"
git lfs pull

echo "==> docker compose build + up"
docker compose -f docker-compose.yml up -d --build

echo "==> done. recent logs (ctrl-c to stop tailing):"
docker compose -f docker-compose.yml logs -f --tail=30 timeline-xr
