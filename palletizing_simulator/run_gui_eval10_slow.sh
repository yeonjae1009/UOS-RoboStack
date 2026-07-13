#!/usr/bin/env bash
# Replay submit_buffer3_search/algorithm_results_eval10 in Isaac Sim GUI.
# Stacking starts automatically after the Isaac Sim window is ready.

ISAAC_PY=/home/robotics/isaac-sim-5.1/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export DISPLAY=:0
GDM_XAUTH="/run/user/$(id -u)/gdm/Xauthority"
[ -f "$GDM_XAUTH" ] && export XAUTHORITY="$GDM_XAUTH"

_strip_ros() { printf '%s' "$1" | tr ':' '\n' | grep -v '/opt/ros/' | paste -sd: -; }
export LD_LIBRARY_PATH="$(_strip_ros "$LD_LIBRARY_PATH")"
export PYTHONPATH="$(_strip_ros "$PYTHONPATH")"
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH \
      ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_LOCALHOST_ONLY \
      RMW_IMPLEMENTATION ROS_DOMAIN_ID LD_PRELOAD 2>/dev/null

LOG="$HERE/sim_gui_eval10_slow.log"
STEP_DELAY="${STEP_DELAY:-1.0}"

echo "[run_gui_eval10_slow] DISPLAY=$DISPLAY  XAUTHORITY=$XAUTHORITY"
echo "[run_gui_eval10_slow] STEP_DELAY=$STEP_DELAY sec/box"

for attempt in 1 2 3 4 5; do
  echo "[run_gui_eval10_slow] attempt $attempt/5 ..."
  "$ISAAC_PY" simulator.py \
    --config config/sim_config_submit_eval10_slow_gui.yaml \
    --step-delay "$STEP_DELAY" \
    2>&1 | tee "$LOG"
  if grep -q "Segmentation fault\|graph.image.core.plugin.*Fatal" "$LOG"; then
    echo "[run_gui_eval10_slow] startup crash detected; retrying..."
    sleep 2
    continue
  fi
  echo "[run_gui_eval10_slow] finished."
  break
done
