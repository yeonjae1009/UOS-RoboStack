#!/usr/bin/env bash
# CJ 대회 스펙(연속, 5종 박스, 1.2x1.0x1.25, 버퍼0, 회전 0/90) PCT 학습 런처.
# 사용:  bash train_cj.sh [실험이름]      (기본 cj_v1)
# 로그:  train_<이름>.log,  체크포인트: logs/experiment/
# 중단:  Ctrl+C (foreground) 또는  pkill -f 'main.py --continuous'
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${1:-cj_v1}"

# ROS 환경 제거(혹시 모를 충돌 방지) — torch venv만 사용
_strip_ros() { printf '%s' "$1" | tr ':' '\n' | grep -v '/opt/ros/' | paste -sd: -; }
export LD_LIBRARY_PATH="$(_strip_ros "$LD_LIBRARY_PATH")"
export PYTHONPATH="$(_strip_ros "$PYTHONPATH")"

echo "$NAME" | .venv-pct/bin/python main.py \
  --continuous --setting 1 \
  --internal-node-holder 200 --leaf-node-holder 100 \
  --num-processes 16 \
  --print-log-interval 20
