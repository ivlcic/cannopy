#!/usr/bin/env bash

set -euo pipefail

DEFAULT_SEED=2611
VRAM_GB=32
TASK_GB=8
MAX_PARALLEL_TASKS=$((VRAM_GB / TASK_GB))
SEED="${1:-}"

if [[ -z "${SEED}" ]]; then
  SEED="${DEFAULT_SEED}"
  echo "Warning: no seed provided, defaulting to ${SEED}." >&2
fi

if (( MAX_PARALLEL_TASKS < 1 )); then
  echo "Error: VRAM_GB=${VRAM_GB} yields no runnable tasks." >&2
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

for run_name in "${RUN_NAMES[@]}"; do
  ./train token ner-sdjt -c mm-bert.yaml \
    -s "data.attributes.run_name=${run_name}" \
    -s "train.seed=${SEED}" &

  while (( $(jobs -pr | wc -l) >= MAX_PARALLEL_TASKS )); do
    wait -n
  done
done

wait
