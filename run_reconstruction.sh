#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"

CONDA_ROOT="${CONDA_ROOT:-/opt/conda}"
RECON_ENV_NAME="${RECON_ENV_NAME:-FabricFlow_reconstruction}"
RECON_PYTHON="${RECON_PYTHON:-$CONDA_ROOT/envs/$RECON_ENV_NAME/bin/python}"
ROTATION_ANGLE="${ROTATION_ANGLE:-95}"

LOG_DIR="$PROJECT_ROOT/reconstruction_logs"
mkdir -p "$LOG_DIR"

RUN_TAG="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_reconstruction_${RUN_TAG}.log"
STATUS_FILE="$LOG_DIR/run_reconstruction_status.txt"
MONITOR_FILE="$LOG_DIR/run_reconstruction_monitor.txt"
PID_FILE="$LOG_DIR/run_reconstruction.pid"

exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

write_status() {
  local stage="$1"
  local message="$2"
  {
    echo "time=$(timestamp)"
    echo "pid=$$"
    echo "stage=$stage"
    echo "message=$message"
    echo "log_file=$LOG_FILE"
  } > "$STATUS_FILE"
  printf '[%s] stage=%s | %s\n' "$(timestamp)" "$stage" "$message" | tee -a "$MONITOR_FILE"
}

CURRENT_STAGE="bootstrap"
trap 'write_status "${CURRENT_STAGE:-unknown}" "FAILED at line $LINENO"' ERR

echo "$$" > "$PID_FILE"
write_status "$CURRENT_STAGE" "starting reconstruction workflow"

[[ -f "$RECON_PYTHON" ]] || { echo "Missing reconstruction Python: $RECON_PYTHON"; exit 1; }
[[ -d "$PROJECT_ROOT/reconstruction/warp_instance_seg_mask" ]] || { echo "Missing reconstruction/warp_instance_seg_mask"; exit 1; }
[[ -d "$PROJECT_ROOT/reconstruction/weft_semantic_seg_mask" ]] || { echo "Missing reconstruction/weft_semantic_seg_mask"; exit 1; }
[[ -d "$PROJECT_ROOT/reconstruction/nnunet_prediction" ]] || { echo "Missing reconstruction/nnunet_prediction"; exit 1; }

CURRENT_STAGE="validate-runtime"
write_status "$CURRENT_STAGE" "checking reconstruction environment"
"$RECON_PYTHON" -c "import cv2, numpy, skimage; print('[ok] reconstruction runtime')"

CURRENT_STAGE="run-reconstruction"
write_status "$CURRENT_STAGE" "running reconstruction/pipeline.py"
"$RECON_PYTHON" "$PROJECT_ROOT/reconstruction/pipeline.py" \
  --base-dir "$PROJECT_ROOT/reconstruction" \
  --rotation-angle "$ROTATION_ANGLE"

CURRENT_STAGE="done"
write_status "$CURRENT_STAGE" "reconstruction workflow finished successfully"
echo "[done] outputs in $PROJECT_ROOT/reconstruction/warp_instance_reconstruction_values and $PROJECT_ROOT/reconstruction/segmentation_results"
