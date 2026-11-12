#!/usr/bin/env bash
# Creates inference/sglang/.venv with SGLang 0.5.23 (the version this plugin is tested with). The plugin itself is not
# installed: launch.sh puts this directory on PYTHONPATH.
#   bash inference/sglang/install.sh
DIR=$(cd "$(dirname "$0")" && pwd) || exit 1
uv venv --python 3.11 "$DIR/.venv" || exit 1
uv pip install --python "$DIR/.venv/bin/python" --prerelease=allow "sglang==0.5.23" || exit 1
"$DIR/.venv/bin/python" -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__)"
