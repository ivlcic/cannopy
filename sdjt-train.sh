#!/usr/bin/env bash

set -euo pipefail

DEFAULT_SEED=2611
VRAM_GB=32
TASK_GB=8
MAX_PARALLEL_TASKS=$((VRAM_GB / TASK_GB))
SESSION_NAME="sdjt-train-s${DEFAULT_SEED}"
SEED="${1:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"

if [[ -z "${SEED}" ]]; then
  SEED="${DEFAULT_SEED}"
  echo "Warning: no seed provided, defaulting to ${SEED}." >&2
fi

SESSION_NAME="sdjt-train-s${SEED}"

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

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Error: tmux session ${SESSION_NAME} already exists." >&2
  exit 1
fi

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
  WINDOW_COMMANDS[worker]+=" && ./train token ner-sdjt -c mm-bert.yaml"
  WINDOW_COMMANDS[worker]+=" -s \"data.attributes.run_name=${run_name}\""
  WINDOW_COMMANDS[worker]+=" -s \"train.seed=${SEED}\""
done

tmux new-session -d -s "${SESSION_NAME}" -n "worker-1"
tmux set-option -t "${SESSION_NAME}" remain-on-exit on

for ((worker=1; worker<MAX_PARALLEL_TASKS; worker++)); do
  tmux new-window -t "${SESSION_NAME}" -n "worker-$((worker + 1))"
done

for ((worker=0; worker<MAX_PARALLEL_TASKS; worker++)); do
  window_name="worker-$((worker + 1))"
  window_command="${WINDOW_COMMANDS[worker]}"
  window_command+="; echo; echo \"${window_name} finished.\""
  printf -v quoted_command "%q" "${window_command}"
  tmux send-keys -t "${SESSION_NAME}:${window_name}" "bash -lc ${quoted_command}" C-m
done

echo "Started ${#RUN_NAMES[@]} runs across ${MAX_PARALLEL_TASKS} tmux windows in session ${SESSION_NAME}."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
