#!/usr/bin/env bash
# Starts an OpenAI-compatible SGLang server for an LLM-jp-VL checkpoint.
#   bash inference/sglang/launch.sh [MODEL] [PORT] [extra sglang.launch_server args...]
#   bash inference/sglang/launch.sh llm-jp/llm-jp-4-vl-9b 30000 --dp-size 8
# Environment: SGLANG_PYTHON (default inference/sglang/.venv/bin/python, see install.sh).
# Note: use the cuDNN bundled with torch; loading an older system cuDNN module breaks the SigLIP convolution
# (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED).
DIR=$(cd "$(dirname "$0")" && pwd) || exit 1
MODEL=${1:-llm-jp/llm-jp-4-vl-9b}
PORT=${2:-30000}
[ $# -gt 0 ] && shift
[ $# -gt 0 ] && shift
PY=${SGLANG_PYTHON:-$DIR/.venv/bin/python}
export PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_EXTERNAL_MODEL_PACKAGE=llmjpvl_sglang.models
exec "$PY" -m sglang.launch_server --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" --trust-remote-code "$@"
