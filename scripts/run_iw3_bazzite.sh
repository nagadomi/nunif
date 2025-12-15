#!/usr/bin/env bash
set -euo pipefail
# Host launcher for iw3 GUI on Bazzite using the distrobox/venv set up by
# scripts/install_bazzite.sh. Runs against host-visible paths.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BOX_NAME="${NUNIF_BAZZITE_CONTAINER:-nunif-bazzite}"
VENV_DIR="${NUNIF_BAZZITE_VENV:-$HOME/.local/share/nunif-bazzite/venv}"
DATA_DIR="${NUNIF_DATA_DIR:-${NUNIF_HOME:-$HOME/nunif_data}}"

if ! command -v distrobox >/dev/null 2>&1; then
  echo "distrobox not found. Run scripts/install_bazzite.sh first." >&2
  exit 1
fi

export NUNIF_HOME="${NUNIF_HOME:-${DATA_DIR}}"

exec distrobox enter "${BOX_NAME}" -- /bin/bash -lc "cd '${REPO_ROOT}' && source '${VENV_DIR}/bin/activate' && NUNIF_HOME='${NUNIF_HOME}' python -m iw3.gui \"\$@\""
