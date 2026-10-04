#!/usr/bin/env bash
# The Phase 0 gate, shrunk to something that runs in seconds.
#
# Same code path as the 1000-frame profile: same camera, rates, encoder and verification.
# Only the frame count differs, so a regression in any of those fails here too. The
# profiler exits non-zero if any of its own checks fail, which is what makes this a gate
# rather than a smoke screen.
#
#   bash scripts/ci_smoke.sh          # uses .venv/bin/python if present, else python3
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi

echo "python: $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"

OUT_DIR="${CI_SMOKE_DIR:-results/env_profile_ci}"

"$PY" -m dronevla.profile_env \
    --frames 10 \
    --frames-per-episode 5 \
    --warmup-frames 2 \
    --renderer tiny \
    --out "${OUT_DIR}/env.json" \
    --frames-dir "${OUT_DIR}/frames" \
    --meta-out "${OUT_DIR}/frames.jsonl" \
    --label "ci smoke"

# profile_env already failed the process on a failed check; re-read the report so the gate
# does not depend on its exit code alone.
"$PY" - "${OUT_DIR}/env.json" <<'PYCHECK'
import json, sys
report = json.load(open(sys.argv[1]))
checks = report["checks"]
bad = [k for k, v in checks.items() if isinstance(v, dict) and v.get("pass") is False]
print(f"re-read {sys.argv[1]}: {len(checks)} checks, {len(bad)} failed")
if bad:
    sys.exit("failed: " + ", ".join(bad))
if report["config"]["physics_steps_per_recorded_frame"] != 48:
    sys.exit("rate contract broken: expected 48 physics steps per recorded frame")
if report["camera"]["width"] != 128 or report["camera"]["height"] != 96:
    sys.exit("observation contract broken: expected a 128x96 frame")
if report["camera"]["channels"] != 3:
    sys.exit("observation contract broken: alpha must be dropped")
print("gate passed")
PYCHECK
