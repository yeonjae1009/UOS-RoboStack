#!/usr/bin/env bash
# 최신 PCT 체크포인트로 대회 두 시퀀스를 적재 → 대회 JSON 변환 → Isaac 입력 폴더로 복사.
# 그 뒤  palletizing_simulator/run_gui.sh  로 Isaac Sim GUI에서 볼 수 있다.
#   사용: bash export_and_view.sh
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=.venv-pct/bin/python
SIM_DIR="../palletizing_simulator/algorithm_results"
SEQ_DIR="../templete code/box_sequence"

CKPT=$(ls -t logs/experiment/cj_v1*/PCT-*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "체크포인트 없음 (logs/experiment/cj_v1*/)"; exit 1; fi
echo "[export] 체크포인트: $CKPT"

for name in box_sequence_0 box_sequence_1; do
  $PY export_to_contest.py \
    --model-path "$CKPT" \
    --box-sequence "$SEQ_DIR/$name.json" \
    --out "out_$name.json" \
    --internal-node-holder 200 --leaf-node-holder 100 \
    2>/dev/null | grep -E "placed|saved"
  cp "out_$name.json" "$SIM_DIR/$name.json"
done
echo "[export] Isaac 입력 폴더로 복사 완료 → $SIM_DIR"
echo "이제:  cd ../palletizing_simulator && bash run_gui.sh   (헤드리스 점수는: .../bin/python simulator.py)"
