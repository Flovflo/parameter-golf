#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runpod_dir="${RUNPOD_STATE_DIR:-$repo_root/.runpod}"
ssh_key_path="${RUNPOD_SSH_KEY_PATH:-$runpod_dir/id_ed25519}"
ssh_pub_path="${ssh_key_path}.pub"
template_id="${RUNPOD_TEMPLATE_ID:-y5cejece4j}"
repo_url="${RUNPOD_REPO_URL:-https://github.com/Flovflo/parameter-golf.git}"

usage() {
  cat <<'EOF'
Usage:
  scripts/runpod.sh setup
  scripts/runpod.sh create [pod-name]
  scripts/runpod.sh smoke [repo-dir]
  scripts/runpod.sh submit [repo-dir]

Env:
  RUNPOD_API_KEY         API key used by runpodctl if not already configured.
  RUNPOD_GPU_ID          GPU id from `runpodctl gpu list` when creating a pod.
  RUNPOD_GPU_COUNT       GPU count used when creating a pod. Defaults to 1.
  RUNPOD_CLOUD_TYPE      Cloud type (SECURE or COMMUNITY). Defaults to SECURE.
  RUNPOD_DATA_CENTER_IDS Comma-separated datacenter preferences for pod creation.
  RUNPOD_TEMPLATE_ID     Template id to deploy. Defaults to the official Parameter Golf template.
  RUNPOD_REPO_URL        Repository cloned by the smoke command.
  RUNPOD_TRAIN_SHARDS    Training shards used by smoke. Defaults to 1.
  RUNPOD_ITERATIONS      Training iterations used by smoke. Defaults to 20.
  RUNPOD_NPROC_PER_NODE  GPU processes used by submit. Defaults to detected GPU count or 1.
EOF
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

extract_pod_id() {
  python3 -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
except json.JSONDecodeError:
    raise SystemExit(1)

stack = [payload]
while stack:
    node = stack.pop()
    if isinstance(node, dict):
        value = node.get("id")
        if isinstance(value, str) and value:
            print(value)
            raise SystemExit(0)
        stack.extend(node.values())
    elif isinstance(node, list):
        stack.extend(node)

raise SystemExit(1)
'
}

has_runpod_credentials() {
  [[ -n "${RUNPOD_API_KEY:-}" || -f "$HOME/.runpod/config.toml" ]]
}

ensure_ssh_key() {
  mkdir -p "$runpod_dir"
  chmod 700 "$runpod_dir"

  if [[ -f "$ssh_key_path" && ! -f "$ssh_pub_path" ]]; then
    ssh-keygen -y -f "$ssh_key_path" >"$ssh_pub_path"
  elif [[ ! -f "$ssh_key_path" && -f "$ssh_pub_path" ]]; then
    echo "Found a public key without a private key at $ssh_pub_path" >&2
    exit 1
  elif [[ ! -f "$ssh_key_path" && ! -f "$ssh_pub_path" ]]; then
    ssh-keygen -t ed25519 -N "" -C "parameter-golf-runpod" -f "$ssh_key_path" >/dev/null
  fi
}

setup_runpod() {
  require_command runpodctl
  require_command ssh-keygen
  ensure_ssh_key

  if has_runpod_credentials; then
    if ! runpodctl ssh add-key --key-file "$ssh_pub_path" >/dev/null; then
      echo "SSH key upload failed or the key already exists. Check `runpodctl ssh list-keys` if needed." >&2
    fi
  else
    echo "Runpod API key not found. Export RUNPOD_API_KEY or run `runpodctl doctor` first, then rerun:" >&2
    echo "  RUNPOD_API_KEY=... scripts/runpod.sh setup" >&2
  fi

  echo "SSH key ready:"
  echo "  private: $ssh_key_path"
  echo "  public : $ssh_pub_path"
}

create_pod() {
  require_command runpodctl
  require_command python3

  if ! has_runpod_credentials; then
    echo "Runpod API key not found. Export RUNPOD_API_KEY or run `runpodctl doctor` first." >&2
    exit 1
  fi

  gpu_id="${RUNPOD_GPU_ID:-}"
  if [[ -z "$gpu_id" ]]; then
    echo "Set RUNPOD_GPU_ID to one of the ids returned by `runpodctl gpu list`." >&2
    exit 1
  fi

  pod_name="${1:-parameter-golf-smoke}"
  create_args=(
    pod create
    --name "$pod_name"
    --template-id "$template_id"
    --gpu-id "$gpu_id"
    --gpu-count "${RUNPOD_GPU_COUNT:-1}"
    --cloud-type "${RUNPOD_CLOUD_TYPE:-SECURE}"
    --volume-in-gb "${RUNPOD_VOLUME_GB:-100}"
    --container-disk-in-gb "${RUNPOD_CONTAINER_DISK_GB:-40}"
    --ssh
    --output json
  )
  if [[ -n "${RUNPOD_DATA_CENTER_IDS:-}" ]]; then
    create_args+=(--data-center-ids "$RUNPOD_DATA_CENTER_IDS")
  fi

  pod_json="$(runpodctl "${create_args[@]}")"

  printf '%s\n' "$pod_json"

  if pod_id="$(printf '%s\n' "$pod_json" | extract_pod_id)"; then
    echo
    echo "SSH details for pod $pod_id:"
    runpodctl ssh info "$pod_id" -v
  fi
}

smoke_test() {
  require_command git
  require_command python3
  require_command torchrun

  repo_dir="${1:-/workspace/parameter-golf}"
  train_shards="${RUNPOD_TRAIN_SHARDS:-1}"
  iterations="${RUNPOD_ITERATIONS:-20}"

  if [[ -d "$repo_dir" && ! -d "$repo_dir/.git" ]]; then
    echo "Repo directory exists but is not a git checkout: $repo_dir" >&2
    exit 1
  fi

  if [[ ! -d "$repo_dir/.git" ]]; then
    git clone "$repo_url" "$repo_dir"
  fi

  cd "$repo_dir"
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$train_shards"

  RUN_ID="${RUNPOD_RUN_ID:-runpod_smoke}" \
  ITERATIONS="$iterations" \
  MAX_WALLCLOCK_SECONDS="${RUNPOD_MAX_WALLCLOCK_SECONDS:-0}" \
  TRAIN_BATCH_TOKENS="${RUNPOD_TRAIN_BATCH_TOKENS:-8192}" \
  TRAIN_LOG_EVERY="${RUNPOD_TRAIN_LOG_EVERY:-5}" \
  VAL_BATCH_SIZE="${RUNPOD_VAL_BATCH_SIZE:-8192}" \
  VAL_LOSS_EVERY="${RUNPOD_VAL_LOSS_EVERY:-0}" \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
}

detect_nproc_per_node() {
  if [[ -n "${RUNPOD_NPROC_PER_NODE:-}" ]]; then
    printf '%s\n' "$RUNPOD_NPROC_PER_NODE"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l | tr -d '[:space:]'
    return
  fi
  printf '1\n'
}

submission_run() {
  require_command git
  require_command python3
  require_command torchrun

  repo_dir="${1:-/workspace/parameter-golf}"
  train_shards="${RUNPOD_TRAIN_SHARDS:-80}"
  nproc_per_node="$(detect_nproc_per_node)"

  if [[ -d "$repo_dir" && ! -d "$repo_dir/.git" ]]; then
    echo "Repo directory exists but is not a git checkout: $repo_dir" >&2
    exit 1
  fi

  if [[ ! -d "$repo_dir/.git" ]]; then
    git clone "$repo_url" "$repo_dir"
  fi

  cd "$repo_dir"
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$train_shards"

  RUN_ID="${RUNPOD_RUN_ID:-runpod_submission}" \
  ITERATIONS="${RUNPOD_ITERATIONS:-9000}" \
  MAX_WALLCLOCK_SECONDS="${RUNPOD_MAX_WALLCLOCK_SECONDS:-600}" \
  TRAIN_BATCH_TOKENS="${RUNPOD_TRAIN_BATCH_TOKENS:-786432}" \
  TRAIN_LOG_EVERY="${RUNPOD_TRAIN_LOG_EVERY:-200}" \
  VAL_BATCH_SIZE="${RUNPOD_VAL_BATCH_SIZE:-524288}" \
  VAL_LOSS_EVERY="${RUNPOD_VAL_LOSS_EVERY:-1000}" \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node="$nproc_per_node" records/submission/train_gpt_submit.py
}

main() {
  case "${1:-}" in
    setup)
      setup_runpod
      ;;
    create)
      shift
      create_pod "${1:-}"
      ;;
    smoke)
      shift
      smoke_test "${1:-}"
      ;;
    submit)
      shift
      submission_run "${1:-}"
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      echo "Unknown command: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
