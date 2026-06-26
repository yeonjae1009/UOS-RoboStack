#!/usr/bin/env bash
# One-shot training summary for the overnight GAT run.
# Live refresh:  watch -n 30 bash isaaclab_pallet/scripts/watch_training.sh
set -u
cd "$(dirname "$0")/../.."
RUN_DIR="${1:-isaaclab_pallet/runs/overnight_full}"
LOG="$RUN_DIR/train.log"

echo "===== [$(basename "$RUN_DIR")] 학습 현황  $(date '+%m-%d %H:%M:%S') ====="
if [ ! -f "$LOG" ]; then
  echo "아직 로그 없음: $LOG  (학습이 시작됐는지 확인하세요)"; exit 0
fi

echo "--- 최근 update 5개 (loss / return / 평균길이) ---"
grep "gat-train] update=" "$LOG" | tail -5 \
  | sed -E 's/.*(update=[0-9]+).*(fps=[0-9.]+).*(loss=[0-9.-]+).*(entropy=[0-9.-]+).*(mean_return=[0-9.-]+).*(mean_len=[0-9.]+).*/  \1  \2  \3  \4  \5  \6/'

echo "--- mean_return 추세 (처음→지금) ---"
rets=$(grep -oE "mean_return=[0-9.-]+" "$LOG" | cut -d= -f2)
if [ -n "$rets" ]; then
  first=$(echo "$rets" | grep -v '^0\.000$' | head -1)
  last=$(echo "$rets" | tail -1)
  best=$(echo "$rets" | sort -g | tail -1)
  echo "  처음(학습초기): ${first:-?}   현재: ${last:-?}   최고: ${best:-?}"
fi

echo "--- 진행/체크포인트 ---"
last_update=$(grep -oE "update=[0-9]+" "$LOG" | tail -1 | cut -d= -f2)
snaps=$(ls "$RUN_DIR"/PCT-update-*.pt 2>/dev/null | wc -l)
latest_ckpt=$(ls -t "$RUN_DIR"/PCT-update-*.pt 2>/dev/null | head -1)
echo "  최신 update: ${last_update:-0}   스냅샷: ${snaps}개   최신: ${latest_ckpt:-없음}"
echo "  resume용: $RUN_DIR/PCT-latest.pt  ($( [ -f "$RUN_DIR/PCT-latest.pt" ] && date -r "$RUN_DIR/PCT-latest.pt" '+%H:%M:%S' || echo 없음 ))"

echo "--- 안정성 ---"
restarts=$(grep -c "restart #" "$LOG" 2>/dev/null); restarts=${restarts:-0}
running=$(pgrep -fc "train_pallet_gat.py" 2>/dev/null); running=${running:-0}
stop_flag=$([ -f "$RUN_DIR/STOP" ] && echo "STOP 걸림" || echo "정상")
echo "  자동 재시작: ${restarts}회   학습 프로세스: ${running}개   상태: ${stop_flag}"

echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /'
