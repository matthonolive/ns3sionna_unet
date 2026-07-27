#!/usr/bin/env bash
# ==================================================================
# serve_validation.sh  (SERVER-SIDE terminal)
# ------------------------------------------------------------------
# Starts the propagation server in the FOREGROUND for exactly one
# ns-3 run (--single_run), configured for the requested model leg.
# Run this first, wait for "Sionna server socket ready ...", then run
# the matching leg with run_ns3_validation.sh in the other terminal.
# The server exits by itself when the ns-3 run closes the session;
# then start it again for the next leg.
#
# Usage:
#   ./serve_validation.sh <scene_dir> <rt|unet|cost231> [cov_thresh]
#
#   ./serve_validation.sh worldbuilding/valsuite/v02_single_wall rt
#   ./serve_validation.sh worldbuilding/valsuite/v02_single_wall unet
#   ./serve_validation.sh worldbuilding/valsuite/v04_sealed_pocket unet 0.3
#   ./serve_validation.sh worldbuilding/valsuite/v02_single_wall cost231
#
# (friis needs no server -- go straight to run_ns3_validation.sh.)
#
# Run from wherever SERVER_PY resolves; activate the server venv first.
# ==================================================================
set -euo pipefail

# ---- EDIT for your checkout ----
SERVER_PY="${SERVER_PY:-contrib/sionna/model/ns3sionna/ns3unet_spectrum.py}"
UNET_RUN="${UNET_RUN:-/home/matth/sionna_developer/ns-allinone-3.40/ns-3.40/contrib/sionna/model/ns3sionna/unet_shannon}"   # model.pt/meta.json/norm_stats.npz
UNET_DEVICE="${UNET_DEVICE:-cuda:0}"
# --------------------------------

SCENE_DIR="$(realpath "$1")"
LEG="$2"
COV_THRESH="${3:-0.5}"

ENV_ROOT="$(dirname "$SCENE_DIR")"

EXTRA=""
case "$LEG" in
    rt)      EXTRA="" ;;
    unet)    EXTRA="--use_unet --unet_run $UNET_RUN --unet_device $UNET_DEVICE --unet_cov_thresh $COV_THRESH" ;;
    cost231) EXTRA="--use_cost231" ;;
    *) echo "unknown leg '$LEG' (rt|unet|cost231)"; exit 1 ;;
esac

echo "[server] scene root : $ENV_ROOT"
echo "[server] leg        : $LEG"
[[ "$LEG" == "unet" ]] && echo "[server] cov_thresh : $COV_THRESH  run: $UNET_RUN"
echo "[server] starting (single run; exits when ns-3 finishes) ..."
echo "-------------------------------------------------------------------"

exec python "$SERVER_PY" --model_folder "$ENV_ROOT" --single_run --est_csi $EXTRA