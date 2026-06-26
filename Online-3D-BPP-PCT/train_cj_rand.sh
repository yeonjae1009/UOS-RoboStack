#!/usr/bin/env bash
# 사이즈 + 무게 "완전 랜덤" 학습 (일반화). 숨겨진 테스트가 5종과 다를 수 있을 때 견고.
#   - setting 3  : 밀도(무게)를 입력으로 받음, 회전 0/90 (orientation=2)
#   - --sample-from-distribution : 박스 사이즈를 U(a,b)에서 매번 랜덤 생성
#   - 밀도(무게)도 매 박스 U(0,1) 랜덤 (bin3D.py 가 처리)
# 사이즈 범위(--sample-left/right-bound)는 대회 박스(0.134~0.315)를 덮도록 0.13~0.33 기본.
#
# 사용:  bash train_cj_rand.sh [실험이름]   (기본 cjrand_v1)
# 중단:  pkill -f 'main.py --continuous'
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${1:-cjrand_v1}"

_strip_ros() { printf '%s' "$1" | tr ':' '\n' | grep -v '/opt/ros/' | paste -sd: -; }
export LD_LIBRARY_PATH="$(_strip_ros "$LD_LIBRARY_PATH")"
export PYTHONPATH="$(_strip_ros "$PYTHONPATH")"

echo "$NAME" | .venv-pct/bin/python main.py \
  --continuous --setting 3 --sample-from-distribution \
  --sample-left-bound 0.13 --sample-right-bound 0.33 \
  --internal-node-holder 200 --leaf-node-holder 100 \
  --num-processes 16 --print-log-interval 20
