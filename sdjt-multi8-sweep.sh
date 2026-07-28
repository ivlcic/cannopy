#!/usr/bin/env bash

set -euo pipefail

DEFAULT_SEED=2611
VRAM_GB=32
TASK_GB=10
MAX_PARALLEL_TASKS=$((VRAM_GB / TASK_GB))
SEED="${1:-}"
REQUESTED_SESSION_NAME="${2:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"

if [[ -z "${SEED}" ]]; then
  SEED="${DEFAULT_SEED}"
  echo "Warning: no seed provided, defaulting to ${SEED}." >&2
fi

SESSION_NAME="${REQUESTED_SESSION_NAME:-sdjt-multi8-sweep-s${SEED}}"
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

TARGET_SESSION="${SESSION_NAME}"
CREATE_NEW_SESSION=1
if [[ -n "${TMUX:-}" ]]; then
  TARGET_SESSION="$(tmux display-message -p '#S')"
  CREATE_NEW_SESSION=0
  if [[ -n "${REQUESTED_SESSION_NAME}" && "${REQUESTED_SESSION_NAME}" != "${TARGET_SESSION}" ]]; then
    echo "Warning: ignoring requested session ${REQUESTED_SESSION_NAME} and using current tmux session ${TARGET_SESSION}." >&2
  fi
elif tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Error: tmux session ${SESSION_NAME} already exists." >&2
  exit 1
fi

run_data_pipeline

# BERTić's published tuning range is 9e-6 to 1e-4; its released
# ELECTRA configuration uses classifier dropout 0.10.
SWEEP_SPECS=(
  "mdeberta3 1.0e-5 0.10"
  "mdeberta3 1.0e-5 0.15"
  "mdeberta3 1.5e-5 0.10"
  "mdeberta3 1.5e-5 0.15"
  "mdeberta3 2.0e-5 0.10"
  "mdeberta3 2.0e-5 0.15"
  "mdeberta3 2.5e-5 0.10"
  "mdeberta3 2.5e-5 0.15"
  "bertic 1.0e-5 0.10"
  "bertic 3.0e-5 0.10"
  "bertic 5.0e-5 0.10"
  "bertic 1.0e-4 0.10"
  "mm-bert 1.0e-5 0.05"
  "mm-bert 1.0e-5 0.10"
  "mm-bert 2.0e-5 0.05"
  "mm-bert 2.0e-5 0.10"
  "mm-bert 3.0e-5 0.05"
  "mm-bert 3.0e-5 0.10"
  "xlmr 1.0e-5 0.10"
  "xlmr 1.0e-5 0.15"
  "xlmr 2.0e-5 0.10"
  "xlmr 2.0e-5 0.15"
  "xlmr 3.0e-5 0.10"
  "xlmr 3.0e-5 0.15"
)

declare -a WINDOW_COMMANDS=()

for ((worker=0; worker<MAX_PARALLEL_TASKS; worker++)); do
  WINDOW_COMMANDS[worker]="cd \"${PROJECT_ROOT}\" && source \"${VENV_ACTIVATE}\""
done

for idx in "${!SWEEP_SPECS[@]}"; do
  read -r model_config learning_rate classifier_dropout <<<"${SWEEP_SPECS[idx]}"
  worker=$((idx % MAX_PARALLEL_TASKS))
  WINDOW_COMMANDS[worker]+=" && ./train token ner-sdjt -c \"${model_config}\""
  WINDOW_COMMANDS[worker]+=" -s \"data.attributes.run_name=multi8\""
  WINDOW_COMMANDS[worker]+=" -s \"train.seed=${SEED}\""
  WINDOW_COMMANDS[worker]+=" -s \"train.learning_rate=${learning_rate}\""
  WINDOW_COMMANDS[worker]+=" -s \"model.classifier_dropout=${classifier_dropout}\""
  WINDOW_COMMANDS[worker]+=" && ./eval token ner-sdjt -c \"${model_config}\""
  WINDOW_COMMANDS[worker]+=" -s \"data.attributes.run_name=multi8\""
  WINDOW_COMMANDS[worker]+=" -s \"train.seed=${SEED}\""
  WINDOW_COMMANDS[worker]+=" -s \"train.learning_rate=${learning_rate}\""
  WINDOW_COMMANDS[worker]+=" -s \"model.classifier_dropout=${classifier_dropout}\""
done

for ((worker=0; worker<MAX_PARALLEL_TASKS; worker++)); do
  window_name="${WINDOW_PREFIX}-worker-$((worker + 1))"
  if tmux list-windows -t "${TARGET_SESSION}" -F '#W' | rg -Fxq "${window_name}"; then
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

echo "Started ${#SWEEP_SPECS[@]} Multi-8 sweep runs across ${MAX_PARALLEL_TASKS} tmux windows in session ${TARGET_SESSION}."
if (( CREATE_NEW_SESSION )); then
  echo "Attach with: tmux attach -t ${TARGET_SESSION}"
else
  echo "Current tmux session: ${TARGET_SESSION}"
fi
