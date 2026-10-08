#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONHOME PYTHONPATH
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export LIBGL_ALWAYS_SOFTWARE=1
export LP_NUM_THREADS=2
export QT_QPA_FONTDIR="${QT_QPA_FONTDIR:-/usr/share/fonts/truetype/dejavu}"
export PYTHONUNBUFFERED=1
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$ROOT/start_page.py" ]]; then
  echo "start_page.py not found next to this script" >&2
  exit 1
fi
# Interpreter selection is independent of the invoking shell's Conda environment.
if [[ "${1:-}" == "--check-env" ]]; then
  exec /usr/bin/python3 -E -u -X utf8 -B "$ROOT/check_environment.py"
fi
if [[ "${1:-}" == "--probe" ]]; then
  /usr/bin/python3 -E -u -X utf8 -B "$ROOT/check_environment.py" --startup
  exec /usr/bin/python3 -E -u -X utf8 -B "$ROOT/manual_control.py" probe
fi
/usr/bin/python3 -E -u -X utf8 -B "$ROOT/check_environment.py" --startup
mkdir -p "$ROOT/runtime"
LOG="$ROOT/runtime/screen_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "屏幕日志：$LOG"
exec /usr/bin/python3 -E -u -X utf8 -B "$ROOT/start_page.py" "$@" > >(tee -a "$LOG") 2>&1
