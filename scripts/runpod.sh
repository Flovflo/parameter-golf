#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
key_path="${RUNPOD_SSH_KEY_PATH:-$HOME/.runpod/ssh/RunPod-Key-Go}"
template_id="${RUNPOD_TEMPLATE_ID:-y5cejece4j}"

usage() {
  cat <<'EOF'
Usage:
  scripts/runpod.sh setup
  scripts/runpod.sh create [name]
  scripts/runpod.sh smoke <pod-id> [remote-dir]
  scripts/runpod.sh delete <pod-id>

Env:
  RUNPOD_GPU_ID           GPU id from `runpodctl gpu list`
  RUNPOD_GPU_COUNT        GPU count, default 1
  RUNPOD_CLOUD_TYPE       SECURE or COMMUNITY, default SECURE
  RUNPOD_PUBLIC_IP        Set to 1 to require a public IP on COMMUNITY
  RUNPOD_TRAIN_SHARDS     Downloaded train shards for smoke, default 1
  RUNPOD_ITERATIONS       Smoke iterations, default 20
EOF
}

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1" >&2; exit 1; }
}

pubkey_json() {
  python3 - <<'PY'
from pathlib import Path
import json, os
pub = Path(os.environ["RUNPOD_SSH_KEY_PATH"]).with_suffix(".pub").read_text().strip()
print(json.dumps({"PUBLIC_KEY": pub, "JUPYTER_PASSWORD": "parameter-golf"}))
PY
}

ssh_info_json() {
  /opt/homebrew/bin/runpodctl ssh info "$1"
}

ssh_port() {
  ssh_info_json "$1" | python3 - <<'PY'
import json, sys
print(json.load(sys.stdin)["port"])
PY
}

ssh_ip() {
  ssh_info_json "$1" | python3 - <<'PY'
import json, sys
print(json.load(sys.stdin)["ip"])
PY
}

setup_cmd() {
  require /opt/homebrew/bin/runpodctl
  /opt/homebrew/bin/runpodctl doctor >/dev/null
  echo "Runpod SSH key ready at $key_path"
}

create_cmd() {
  require /opt/homebrew/bin/runpodctl
  [[ -n "${RUNPOD_GPU_ID:-}" ]] || { echo "Set RUNPOD_GPU_ID first" >&2; exit 1; }
  export RUNPOD_SSH_KEY_PATH="$key_path"
  env_json="$(pubkey_json)"
  args=(
    pod create --name "${1:-parameter-golf-smoke}" --template-id "$template_id"
    --gpu-id "$RUNPOD_GPU_ID" --gpu-count "${RUNPOD_GPU_COUNT:-1}"
    --cloud-type "${RUNPOD_CLOUD_TYPE:-SECURE}" --env "$env_json"
  )
  [[ "${RUNPOD_CLOUD_TYPE:-SECURE}" == "COMMUNITY" && "${RUNPOD_PUBLIC_IP:-0}" == "1" ]] && args+=(--public-ip)
  /opt/homebrew/bin/runpodctl "${args[@]}"
}

smoke_cmd() {
  require rsync
  pod_id="$1"
  remote_dir="${2:-/workspace/parameter-golf}"
  port="$(ssh_port "$pod_id")"
  ip="$(ssh_ip "$pod_id")"
  ssh_base=(ssh -i "$key_path" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$port")
  rsync_base=(rsync -az -e "ssh -i $key_path -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $port")
  "${ssh_base[@]}" "root@$ip" "mkdir -p '$remote_dir/data'"
  "${rsync_base[@]}" "$repo_root/train_gpt.py" "root@$ip:$remote_dir/"
  "${rsync_base[@]}" "$repo_root/data/cached_challenge_fineweb.py" "root@$ip:$remote_dir/data/"
  if [[ -f "$repo_root/data/manifest.json" ]]; then
    "${rsync_base[@]}" "$repo_root/data/manifest.json" "root@$ip:$remote_dir/data/"
  fi
  "${ssh_base[@]}" "root@$ip" "
    cd '$remote_dir' &&
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards '${RUNPOD_TRAIN_SHARDS:-1}' &&
    RUN_ID='${RUNPOD_RUN_ID:-runpod_smoke}' \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    ITERATIONS='${RUNPOD_ITERATIONS:-20}' \
    MAX_WALLCLOCK_SECONDS=0 \
    VAL_LOSS_EVERY=0 \
    TRAIN_LOG_EVERY=5 \
    TRAIN_BATCH_TOKENS='${RUNPOD_TRAIN_BATCH_TOKENS:-8192}' \
    VAL_BATCH_SIZE='${RUNPOD_VAL_BATCH_SIZE:-8192}' \
    torchrun --standalone --nproc_per_node=1 train_gpt.py
  "
}

delete_cmd() {
  /opt/homebrew/bin/runpodctl pod delete "$1"
}

case "${1:-}" in
  setup) setup_cmd ;;
  create) shift; create_cmd "${1:-}" ;;
  smoke) shift; smoke_cmd "${1:?missing pod id}" "${2:-}" ;;
  delete) shift; delete_cmd "${1:?missing pod id}" ;;
  ""|-h|--help|help) usage ;;
  *) echo "Unknown command: $1" >&2; usage >&2; exit 1 ;;
esac
