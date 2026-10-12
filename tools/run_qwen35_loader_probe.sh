#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 ARM OUTPUT_DIR [NUM_THREADS]" >&2
  exit 2
fi

arm=$1
output_dir=$2
num_threads=${3:-4}
model=${STATEAXIS_LOADER_MODEL:-/data/shared_models/Qwen3.5-35B-A3B}
python=${STATEAXIS_LOADER_PYTHON:-/root/vllm-hust/.venv/bin/python}
port=${STATEAXIS_LOADER_PORT:-18195}
timeout_seconds=${STATEAXIS_LOADER_TIMEOUT_SECONDS:-900}

case "$arm" in
  control|multithread|prefetch|numa|sharded) ;;
  *) echo "arm must be control, multithread, prefetch, numa, or sharded" >&2; exit 2 ;;
esac

[[ -d "$model" ]] || { echo "model directory missing: $model" >&2; exit 2; }
[[ -x "$python" ]] || { echo "python missing: $python" >&2; exit 2; }
mkdir "$output_dir"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -KILL -- "-$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

npu-smi info >"$output_dir/npu-before.txt"
start_ns=$(date +%s%N)
printf '%s\n' "$start_ns" >"$output_dir/process-start-ns.txt"

loader_args=()
if [[ "$arm" == multithread ]]; then
  loader_args=(--model-loader-extra-config \
    "{\"enable_multithread_load\":true,\"num_threads\":$num_threads}")
elif [[ "$arm" == prefetch ]]; then
  loader_args=(
    --safetensors-load-strategy prefetch
    --safetensors-prefetch-num-threads "$num_threads"
    --safetensors-prefetch-block-size 64M
  )
elif [[ "$arm" == sharded ]]; then
  loader_args=(--load-format sharded_state)
fi

ep_args=()
if [[ "${STATEAXIS_LOADER_ENABLE_EP:-0}" == 1 ]]; then
  ep_args=(--enable-expert-parallel)
fi

launcher_prefix=()
if [[ "$arm" == numa ]]; then
  launcher_prefix=(numactl --cpunodebind=2 --membind=2)
fi

source /usr/local/Ascend/cann/set_env.sh >/dev/null 2>&1
setsid "${launcher_prefix[@]}" env \
  -u ASCEND_RT_VISIBLE_DEVICES \
  -u ASCEND_VISIBLE_DEVICES \
  VLLM_USE_V1=1 \
  "$python" -m vllm.entrypoints.cli.main serve "$model" \
    --served-model-name qwen35-loader-probe \
    --host 127.0.0.1 \
    --port "$port" \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 2048 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching \
    --enforce-eager \
    --distributed-executor-backend mp \
    --disable-custom-all-reduce \
    --no-enable-log-requests \
    --uvicorn-log-level warning \
    "${ep_args[@]}" \
    "${loader_args[@]}" \
    >"$output_dir/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$output_dir/server.pid"

ready=false
for _ in $(seq 1 $((timeout_seconds / 2))); do
  if curl -fsS "http://127.0.0.1:$port/health" >"$output_dir/health.json" 2>/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid" || true
    echo "server exited before readiness" >&2
    exit 1
  fi
  sleep 2
done
[[ "$ready" == true ]] || { echo "server readiness timed out" >&2; exit 1; }

ready_ns=$(date +%s%N)
printf '%s\n' "$ready_ns" >"$output_dir/ready-ns.txt"
curl -fsS "http://127.0.0.1:$port/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-loader-probe","messages":[{"role":"user","content":"Reply with exactly the word READY."}],"temperature":0,"max_tokens":16}' \
  >"$output_dir/response.json"
response_ns=$(date +%s%N)
printf '%s\n' "$response_ns" >"$output_dir/response-ns.txt"
npu-smi info >"$output_dir/npu-ready.txt"

"$python" - "$arm" "$num_threads" "$model" "$output_dir" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

arm, threads, model, output_dir = sys.argv[1:]
root = pathlib.Path(output_dir)
start = int((root / "process-start-ns.txt").read_text())
ready = int((root / "ready-ns.txt").read_text())
response = int((root / "response-ns.txt").read_text())
payload = json.loads((root / "response.json").read_text())
content = payload["choices"][0]["message"]["content"]

def git(path: str, *args: str) -> str:
    return subprocess.check_output(["git", "-C", path, *args], text=True).strip()

log = (root / "server.log").read_text(errors="replace")
import re

weight_seconds = [float(value) for value in re.findall(
    r"Loading weights took ([0-9.]+) seconds", log
)]
rank_load_seconds = [float(value) for value in re.findall(
    r"load_model total time: ([0-9.]+) seconds", log
)]
result = {
    "schema_version": "stateaxis-qwen35-loader-diagnostic/v1",
    "arm": arm,
    "carrier": {
        "control": "vllm-serial-safetensors",
        "multithread": "vllm-multithread-safetensors",
        "prefetch": "vllm-bounded-page-cache-prefetch",
        "numa": "vllm-serial-safetensors-numa-local",
        "sharded": "vllm-tp-sharded-state",
    }[arm],
    "formal_stateaxis_mod_on": False,
    "num_threads": int(threads) if arm in {"multithread", "prefetch"} else 1,
    "expert_parallel": "enable_expert_parallel: True" in log,
    "model": model,
    "model_config_sha256": hashlib.sha256(
        (pathlib.Path(model) / "config.json").read_bytes()
    ).hexdigest(),
    "model_index_sha256": hashlib.sha256(
        (pathlib.Path(model) / "model.safetensors.index.json").read_bytes()
    ).hexdigest(),
    "vllm_commit": git("/vllm-workspace/vllm", "rev-parse", "HEAD"),
    "vllm_ascend_commit": git("/vllm-workspace/vllm-ascend", "rev-parse", "HEAD"),
    "process_to_ready_seconds": (ready - start) / 1e9,
    "weight_iterator_seconds": weight_seconds,
    "rank_load_model_seconds": rank_load_seconds,
    "first_request_seconds": (response - ready) / 1e9,
    "response_content": content,
    "response_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
    "evidence_label": "real-npu-carrier-diagnostic-not-mod-performance",
}
(root / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
PY

cleanup
server_pid=""
npu-smi info >"$output_dir/npu-after.txt"
