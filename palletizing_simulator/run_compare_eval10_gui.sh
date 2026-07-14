#!/usr/bin/env bash

# Side-by-side GUI replay:
# left  = submit_buffer3_search/algorithm_results_eval10/1400
# right = submit_buffer3_search/algorithm_results_eval10/1450

ISAAC_PY=/home/robotics/isaac-sim-5.1/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export DISPLAY="${DISPLAY:-:0}"
GDM_XAUTH="/run/user/$(id -u)/gdm/Xauthority"
[ -f "$GDM_XAUTH" ] && export XAUTHORITY="$GDM_XAUTH"

strip_ros() {
  printf '%s' "${1:-}" | tr ':' '\n' | grep -v '/opt/ros/' | paste -sd: - || true
}

export LD_LIBRARY_PATH="$(strip_ros "${LD_LIBRARY_PATH:-}")"
export PYTHONPATH="$(strip_ros "${PYTHONPATH:-}")"
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH \
      ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_LOCALHOST_ONLY \
      RMW_IMPLEMENTATION ROS_DOMAIN_ID LD_PRELOAD 2>/dev/null || true

LOG="$HERE/compare_eval10_gui.log"
STEP_DELAY="${STEP_DELAY:-0.4}"
EPISODE_DELAY="${EPISODE_DELAY:-2.0}"

echo "[run_compare_eval10_gui] DISPLAY=$DISPLAY XAUTHORITY=${XAUTHORITY:-}"
echo "[run_compare_eval10_gui] STEP_DELAY=$STEP_DELAY EPISODE_DELAY=$EPISODE_DELAY"
echo "[run_compare_eval10_gui] log=$LOG"

"$ISAAC_PY" compare_replay_gui.py \
  --step-delay "$STEP_DELAY" \
  --episode-delay "$EPISODE_DELAY" \
  --limit 10 \
  "$@" \
  2>&1 | tee "$LOG"
