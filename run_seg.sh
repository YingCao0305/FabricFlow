#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_seg.sh [check|pretrain|finetune-predict|finetune-train|test|full]

Commands:
  check             Validate dataset layout and train/val split
  pretrain          Run nnUNet semantic pretraining and Cellpose pretraining
  finetune-predict  Run nnUNet prediction on finetune_image and stop
  finetune-train    Prepare Cellpose finetune data and run Cellpose finetuning
  test              Run test prediction and export final warp masks
  full              Run the full segmentation workflow
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
COMMAND="${1:-full}"

case "$COMMAND" in
  check|pretrain|finetune-predict|finetune-train|test|full) ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

CONDA_ROOT="${CONDA_ROOT:-/opt/conda}"
SEMANTIC_ENV_NAME="${SEMANTIC_ENV_NAME:-FabricFlow_semantic}"
INSTANCE_ENV_NAME="${INSTANCE_ENV_NAME:-FabricFlow_instance}"

SEMANTIC_PYTHON="${SEMANTIC_PYTHON:-$CONDA_ROOT/envs/$SEMANTIC_ENV_NAME/bin/python}"
INSTANCE_PYTHON="${INSTANCE_PYTHON:-$CONDA_ROOT/envs/$INSTANCE_ENV_NAME/bin/python}"

DATASET_ID="${DATASET_ID:-666}"
DATASET_NAME="${DATASET_NAME:-Dataset666_FiberSegmentation}"
CONFIGURATION="${CONFIGURATION:-2d}"
FOLD="${FOLD:-0}"
SPLIT_INDEX="${SPLIT_INDEX:-0}"

NNUNET_TRAINER="${NNUNET_TRAINER:-nnUNetTrainer}"
CELLPOSE_PRETRAIN_EPOCHS="${CELLPOSE_PRETRAIN_EPOCHS:-500}"
CELLPOSE_FINETUNE_EPOCHS="${CELLPOSE_FINETUNE_EPOCHS:-100}"
CELLPOSE_LR="${CELLPOSE_LR:-0.001}"
CELLPOSE_DIAM_MEAN="${CELLPOSE_DIAM_MEAN:-25}"
CPU_FLAG="${CPU_FLAG:-0}"

DATASET_ROOT="$PROJECT_ROOT/nnUNet-master/DATASET/nnUNet_raw/$DATASET_NAME"
IMAGES_TR="$DATASET_ROOT/imagesTr"
LABELS_TR="$DATASET_ROOT/labelsTr"
FINETUNE_IMAGE="$DATASET_ROOT/finetune_image"
FINETUNE_PRED="$DATASET_ROOT/finetune_prediction"
FINETUNE_REFINE="$DATASET_ROOT/finetune_prediction_manual_refine"
TEST_IMAGE="$DATASET_ROOT/test_image"
DATASET_JSON="$DATASET_ROOT/dataset.json"

LOG_DIR="$PROJECT_ROOT/segmentation_logs"
mkdir -p "$LOG_DIR"

RUN_TAG="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_seg_${RUN_TAG}.log"
STATUS_FILE="$LOG_DIR/run_seg_status.txt"
MONITOR_FILE="$LOG_DIR/run_seg_monitor.txt"
PID_FILE="$LOG_DIR/run_seg.pid"

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

fail_status() {
  local stage="$1"
  local line="$2"
  write_status "$stage" "FAILED at line $line"
}

CURRENT_STAGE="bootstrap"
trap 'fail_status "${CURRENT_STAGE:-unknown}" "$LINENO"' ERR

echo "$$" > "$PID_FILE"
write_status "$CURRENT_STAGE" "starting segmentation workflow"

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "Missing file: $path"; exit 1; }
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || { echo "Missing directory: $path"; exit 1; }
}

count_nii_files() {
  local dir_path="$1"
  find "$dir_path" -maxdepth 1 -type f -name '*.nii.gz' | wc -l
}

ensure_inputs() {
  require_file "$SEMANTIC_PYTHON"
  require_file "$INSTANCE_PYTHON"
  require_file "$DATASET_JSON"
  require_dir "$IMAGES_TR"
  require_dir "$LABELS_TR"
  require_dir "$FINETUNE_IMAGE"
  require_dir "$TEST_IMAGE"
  mkdir -p "$FINETUNE_REFINE"
}

sync_dataset_json() {
  CURRENT_STAGE="sync-dataset-json"
  write_status "$CURRENT_STAGE" "synchronizing dataset.json numTraining"
  "$SEMANTIC_PYTHON" - <<'PY' "$DATASET_JSON"
import json
from pathlib import Path
import sys

dataset_json = Path(sys.argv[1])
images_tr = dataset_json.parent / "imagesTr"
count = len(list(images_tr.glob("*_0000.nii.gz")))
data = json.loads(dataset_json.read_text())
data["numTraining"] = count
dataset_json.write_text(json.dumps(data, indent=4) + "\n")
print(f"[sync] numTraining -> {count}")
PY
}

validate_runtimes() {
  CURRENT_STAGE="validate-runtimes"
  write_status "$CURRENT_STAGE" "checking semantic and instance environments"
  PYTHONPATH="$PROJECT_ROOT/nnUNet-master:$PROJECT_ROOT/cellpose-main" \
    "$SEMANTIC_PYTHON" -c "import nnunetv2, torch, SimpleITK; print('[ok] semantic runtime')"
  PYTHONPATH="$PROJECT_ROOT/nnUNet-master:$PROJECT_ROOT/cellpose-main" \
    "$INSTANCE_PYTHON" -c "import cellpose, torch; print('[ok] instance runtime')"
}

run_pipeline_stage() {
  local stage_name="$1"
  local pipeline_command="$2"
  CURRENT_STAGE="$stage_name"
  write_status "$CURRENT_STAGE" "running segmentation/pipeline.py $pipeline_command"

  local extra_flags=()
  if [[ "$CPU_FLAG" == "1" ]]; then
    extra_flags+=(--cpu)
  fi

  "$SEMANTIC_PYTHON" "$PROJECT_ROOT/segmentation/pipeline.py" "$pipeline_command" \
    --repo-root "$PROJECT_ROOT" \
    --semantic-env "$SEMANTIC_ENV_NAME" \
    --instance-env "$INSTANCE_ENV_NAME" \
    --semantic-python "$SEMANTIC_PYTHON" \
    --instance-python "$INSTANCE_PYTHON" \
    --dataset-id "$DATASET_ID" \
    --dataset-name "$DATASET_NAME" \
    --configuration "$CONFIGURATION" \
    --fold "$FOLD" \
    --split-index "$SPLIT_INDEX" \
    --nnunet-trainer "$NNUNET_TRAINER" \
    --cellpose-pretrain-epochs "$CELLPOSE_PRETRAIN_EPOCHS" \
    --cellpose-finetune-epochs "$CELLPOSE_FINETUNE_EPOCHS" \
    --cellpose-learning-rate "$CELLPOSE_LR" \
    --test-diam-mean "$CELLPOSE_DIAM_MEAN" \
    "${extra_flags[@]}"
}

require_refine_masks() {
  local refine_count
  refine_count="$(count_nii_files "$FINETUNE_REFINE")"
  if [[ "$refine_count" -eq 0 ]]; then
    CURRENT_STAGE="waiting-manual-refine"
    write_status "$CURRENT_STAGE" "no refined masks found; fill finetune_prediction_manual_refine and rerun bash run_seg.sh finetune-train"
    echo "Manual action required:"
    echo "  1. Review predictions in $FINETUNE_PRED"
    echo "  2. Save corrected labels into $FINETUNE_REFINE"
    echo "  3. Rerun: bash run_seg.sh finetune-train"
    exit 0
  fi
}

run_check() {
  sync_dataset_json
  validate_runtimes
  run_pipeline_stage "check" check
}

run_pretrain() {
  run_check
  run_pipeline_stage "pretrain" pretrain
}

run_finetune_predict() {
  run_pipeline_stage "finetune-predict" finetune-predict
}

run_finetune_train() {
  require_refine_masks
  run_pipeline_stage "finetune-prepare" finetune-prepare
  run_pipeline_stage "cellpose-finetune" finetune-train
}

run_test_stage() {
  run_pipeline_stage "test" test
}

ensure_inputs

case "$COMMAND" in
  check)
    run_check
    ;;
  pretrain)
    run_pretrain
    ;;
  finetune-predict)
    run_finetune_predict
    ;;
  finetune-train)
    run_finetune_train
    ;;
  test)
    run_test_stage
    ;;
  full)
    run_pretrain
    run_finetune_predict
    require_refine_masks
    run_finetune_train
    run_test_stage
    ;;
esac

CURRENT_STAGE="done"
write_status "$CURRENT_STAGE" "segmentation workflow finished successfully"
echo "[done] final masks in $PROJECT_ROOT/reconstruction/warp_instance_seg_mask"
