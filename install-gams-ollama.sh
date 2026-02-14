#!/usr/bin/env bash
set -Eeuo pipefail

# ---- config ----
WORKDIR="gams-it-dpo-translator"
MODEL_FILE="GaMS-9B-Instruct-DPO-Translator.Q8_0.gguf"
MODEL_URL="https://huggingface.co/mradermacher/GaMS-9B-Instruct-DPO-Translator-GGUF/resolve/main/GaMS-9B-Instruct-DPO-Translator.Q8_0.gguf"
TAG="gams-it-dpo-translator:9b"
OLLAMA_HOST_DEFAULT="http://127.0.0.1:11434"

# ---- helpers ----
log()  { printf "\n[+] %s\n" "$*"; }
err()  { printf "\n[!] %s\n" "$*" >&2; }
die()  { err "$*"; exit 1; }

cleanup() {
  if [[ -d "$WORKDIR" ]]; then
    rm -rf "$WORKDIR"
  fi
}
trap cleanup EXIT

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"
}

# ---- checks ----
require_cmd mkdir
require_cmd cd
require_cmd rm
require_cmd cat
require_cmd curl
require_cmd ollama

log "Checking Ollama server..."
OLLAMA_HOST="${OLLAMA_HOST:-$OLLAMA_HOST_DEFAULT}"

# Prefer "ollama ps" if available; fallback to HTTP /api/tags
if ollama ps >/dev/null 2>&1; then
  log "Ollama CLI is responsive."
else
  log "Ollama CLI check failed; trying HTTP check at ${OLLAMA_HOST} ..."
  if ! curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    die "Ollama does not appear to be running. Start it, then re-run this script."
  fi
  log "Ollama HTTP endpoint is reachable."
fi

# ---- main ----
log "Creating temp workspace: ${WORKDIR}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

log "Downloading model (this can be large)..."
# -f: fail on HTTP errors, -L: follow redirects, -S: show errors, --retry: basic robustness
curl -fL --retry 5 --retry-delay 2 --retry-all-errors \
  -o "$MODEL_FILE" \
  "$MODEL_URL"

[[ -s "$MODEL_FILE" ]] || die "Download failed or produced an empty file: ${MODEL_FILE}"

log "Writing Modelfile..."
cat > Modelfile <<'EOF'
FROM ./GaMS-9B-Instruct-DPO-Translator.Q8_0.gguf
EOF

log "Creating Ollama model: ${TAG}"
# Capture stderr on failure for easier debugging
if ! ollama create "$TAG" -f Modelfile; then
  die "ollama create failed."
fi

log "Verifying model exists..."
if ! ollama show "$TAG" >/dev/null 2>&1; then
  die "Model was not found after creation: ${TAG}"
fi

log "Success! Model created: ${TAG}"
log "Try: ollama run ${TAG}"
