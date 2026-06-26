#!/usr/bin/env bash
# CJ 대회 스펙 + setting 3 (밀도 반영) PCT 학습 런처.
#
# setting 3 = 박스마다 밀도가 달라, 안정성 계산(mass=부피×밀도 → COM)에 밀도가 반영됨.
# 학습 중에는 밀도를 랜덤 U(0,1)로 샘플링(표준 setting3). 실제 대회 밀도는 추론 때
# 정규화해서 넣는다. 관찰값 내부노드 길이가 6→7로 바뀌므로 setting1 모델과 호환 안 됨
# (처음부터 학습). 회전은 setting1과 동일하게 0/90 (orientation=2).
#
# 사용:  bash train_cj3.sh [실험이름]   (기본 cj3_v1)
# 중단:  pkill -f 'main.py --continuous'
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${1:-cj3_v1}"

_strip_ros() { printf '%s' "$1" | tr ':' '\n' | grep -v '/opt/ros/' | paste -sd: -; }
export LD_LIBRARY_PATH="$(_strip_ros "$LD_LIBRARY_PATH")"
export PYTHONPATH="$(_strip_ros "$PYTHONPATH")"

echo "$NAME" | .venv-pct/bin/python main.py \
  --continuous --setting 3 \
  --internal-node-holder 200 --leaf-node-holder 100 \
  --num-processes 16 \
  --print-log-interval 20
