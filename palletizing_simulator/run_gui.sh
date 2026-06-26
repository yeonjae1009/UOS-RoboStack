#!/usr/bin/env bash
# Isaac Sim GUI(에디터 창)로 팔레타이징 시뮬레이터 실행.
#
# 이 PC에서 풀 에디터 experience는 omni.kit.exec.core / omni.graph.image 플러그인
# 초기화 단계에서 간헐적으로 segfault 한다(시작 ~1초). 일단 부팅을 넘기면 안정적이다.
# 그래서 (1) 검증된 DISPLAY/XAUTHORITY 강제, (2) ROS 환경 제거, (3) 시작 크래시 시
# 자동 재시도 로 부팅 성공률을 끌어올린다.
#
# 사용법:  bash run_gui.sh
# 창이 뜨면 좌측 툴바의 ▶ PLAY 버튼을 눌러 시뮬레이션을 시작한다.

ISAAC_PY=/home/robotics/isaac-sim-5.1/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# (1) 검증된 작동 디스플레이/인증 강제 (세션이 X11 :0, gdm Xauthority인 것 확인됨)
export DISPLAY=:0
GDM_XAUTH="/run/user/$(id -u)/gdm/Xauthority"
[ -f "$GDM_XAUTH" ] && export XAUTHORITY="$GDM_XAUTH"

# (2) ROS 환경 제거 — Isaac 프로세스에 /opt/ros 라이브러리가 섞이지 않도록
_strip_ros() { printf '%s' "$1" | tr ':' '\n' | grep -v '/opt/ros/' | paste -sd: -; }
export LD_LIBRARY_PATH="$(_strip_ros "$LD_LIBRARY_PATH")"
export PYTHONPATH="$(_strip_ros "$PYTHONPATH")"
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH \
      ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_LOCALHOST_ONLY \
      RMW_IMPLEMENTATION ROS_DOMAIN_ID LD_PRELOAD 2>/dev/null

LOG="$HERE/sim_gui.log"
echo "[run_gui] DISPLAY=$DISPLAY  XAUTHORITY=$XAUTHORITY"

# (3) 시작 크래시 자동 재시도 (최대 5회). 부팅 성공("app ready") 시 그대로 진행.
for attempt in 1 2 3 4 5; do
  echo "[run_gui] 시도 $attempt/5 ..."
  "$ISAAC_PY" simulator.py --config config/sim_config_gui.yaml 2>&1 | tee "$LOG"
  if grep -q "Segmentation fault\|graph.image.core.plugin.*Fatal" "$LOG"; then
    echo "[run_gui] 시작 단계 크래시 감지 → 재시도..."
    sleep 2
    continue
  fi
  echo "[run_gui] 정상 종료."
  break
done
