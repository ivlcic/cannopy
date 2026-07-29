#!/usr/bin/env bash

set -euo pipefail

DEFAULT_SEED=2611
VRAM_GB=16
TASK_GB=16
MAX_PARALLEL_TASKS=$((VRAM_GB / TASK_GB))
SEED="${1:-}"
MODEL_NAME="${2:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"

usage() {
  echo "Usage: $0 [seed] <model-name>" >&2
  echo "Example: $0 2611 mm-bert" >&2
}

run_data_pipeline() {
  echo "Running SDJT data pipeline for seed ${SEED}..." >&2
  (
    cd "${PROJECT_ROOT}"
    # shellcheck disable=SC1090
    source "${VENV_ACTIVATE}"
    ./data split ner -s "data.split.seed=${SEED}"
    ./data analyze ner -s "data.split.seed=${SEED}"
    ./data resample ner-sdjt -s "data.split.seed=${SEED}" -s "data.sampling.seed=${SEED}"
    ./data analyze ner-sdjt -s "data.split.seed=${SEED}" -s "data.sampling.seed=${SEED}"
  )
  echo "Finished SDJT data pipeline for seed ${SEED}." >&2
}

if [[ -z "${SEED}" ]]; then
  SEED="${DEFAULT_SEED}"
  echo "Warning: no seed provided, defaulting to ${SEED}." >&2
fi

if [[ -z "${MODEL_NAME}" ]]; then
  echo "Error: no model name provided." >&2
  usage
  exit 1
fi

if [[ ! "${MODEL_NAME}" =~ ^[[:alnum:]][[:alnum:]._-]*$ ]]; then
  echo "Error: invalid model name ${MODEL_NAME}." >&2
  exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/conf/model/${MODEL_NAME}.yaml" ]]; then
  echo "Error: model config conf/model/${MODEL_NAME}.yaml does not exist." >&2
  exit 1
fi

SESSION_NAME="sdjt-train-${MODEL_NAME}-s${SEED}"
WINDOW_PREFIX="${SESSION_NAME}"

if (( MAX_PARALLEL_TASKS < 1 )); then
  echo "Error: VRAM_GB=${VRAM_GB} yields no runnable tasks." >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is not installed." >&2
  exit 1
fi

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Error: virtualenv activation script not found at ${VENV_ACTIVATE}." >&2
  exit 1
fi

TARGET_SESSION="${SESSION_NAME}"
CREATE_NEW_SESSION=1
if [[ -n "${TMUX:-}" ]]; then
  TARGET_SESSION="$(tmux display-message -p '#S')"
  CREATE_NEW_SESSION=0
elif tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Error: tmux session ${SESSION_NAME} already exists." >&2
  exit 1
fi

run_data_pipeline

RUN_NAMES=(
  mono-bg
  mono-cs
  mono-hr
  mono-pl
  mono-ru
  mono-sl
  mono-sr
  mono-uk
  multi8
  multi12
  full-multi8
  full-multi12
  full-multi12-capaux
  multi7-no-hr
  multi7-plus-hr500k
  multi7-plus-hr-wikiann
  multi8-full-bg
  multi8-full-cs
  multi8-full-hr
  multi8-full-pl
  multi8-full-ru
  multi8-full-sl
  multi8-full-sr
  multi8-full-uk
  pretrain-multi7-full-bg
  pretrain-multi7-full-cs
  pretrain-multi7-full-hr
  pretrain-multi7-full-pl
  pretrain-multi7-full-ru
  pretrain-multi7-full-sl
  pretrain-multi7-full-sr
  pretrain-multi7-full-uk
  mono-sl-p10
  mono-sl-p25
  mono-sl-p50
  multi8-sl-p10
  multi8-sl-p25
  multi8-sl-p50
  multi12-sl-p10
  multi12-sl-p25
  multi12-sl-p50
  mono-sr-p10
  mono-sr-p25
  mono-sr-p50
  multi8-sr-p10
  multi8-sr-p25
  multi8-sr-p50
  multi12-sr-p10
  multi12-sr-p25
  multi12-sr-p50
)

declare -a WINDOW_COMMANDS=()

for ((worker=0; worker<MAX_PARALLEL_TASKS; worker++)); do
  WINDOW_COMMANDS[worker]="cd \"${PROJECT_ROOT}\" && source \"${VENV_ACTIVATE}\""
done

for idx in "${!RUN_NAMES[@]}"; do
  run_name="${RUN_NAMES[idx]}"
  worker=$((idx % MAX_PARALLEL_TASKS))
  WINDOW_COMMANDS[worker]+=" && ./train token ner-sdjt -c \"${MODEL_NAME}\""
  WINDOW_COMMANDS[worker]+=" -s \"data.attributes.run_name=${run_name}\""
  WINDOW_COMMANDS[worker]+=" -s \"train.seed=${SEED}\""
done

for ((worker=0; worker<MAX_PARALLEL_TASKS; worker++)); do
  window_name="${WINDOW_PREFIX}-worker-$((worker + 1))"
  if tmux list-windows -t "${TARGET_SESSION}" -F '#W' | grep -Fxq "${window_name}"; then
    echo "Error: tmux window ${window_name} already exists in session ${TARGET_SESSION}." >&2
    exit 1
  fi
done

first_window_name="${WINDOW_PREFIX}-worker-1"
if (( CREATE_NEW_SESSION )); then
  tmux new-session -d -s "${TARGET_SESSION}" -n "${first_window_name}"
  tmux setw -t "${TARGET_SESSION}:${first_window_name}" remain-on-exit on
else
  tmux new-window -t "${TARGET_SESSION}" -n "${first_window_name}"
  tmux setw -t "${TARGET_SESSION}:${first_window_name}" remain-on-exit on
fi

for ((worker=1; worker<MAX_PARALLEL_TASKS; worker++)); do
  window_name="${WINDOW_PREFIX}-worker-$((worker + 1))"
  tmux new-window -t "${TARGET_SESSION}" -n "${window_name}"
  tmux setw -t "${TARGET_SESSION}:${window_name}" remain-on-exit on
done

for ((worker=0; worker<MAX_PARALLEL_TASKS; worker++)); do
  window_name="${WINDOW_PREFIX}-worker-$((worker + 1))"
  window_command="${WINDOW_COMMANDS[worker]}"
  window_command+="; echo; echo \"${window_name} finished.\""
  printf -v quoted_command "%q" "${window_command}"
  tmux send-keys -t "${TARGET_SESSION}:${window_name}" "bash -lc ${quoted_command}" C-m
done

echo "Started ${#RUN_NAMES[@]} runs across ${MAX_PARALLEL_TASKS} tmux windows in session ${TARGET_SESSION}."
if (( CREATE_NEW_SESSION )); then
  echo "Attach with: tmux attach -t ${TARGET_SESSION}"
else
  echo "Current tmux session: ${TARGET_SESSION}"
fi
