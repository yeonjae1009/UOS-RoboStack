#!/usr/bin/env bash
# PCT 학습 진행 모니터 (실행 자동 감지 + 실시간 runinfo 기준).  사용: bash monitor.sh
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==================== PCT 학습 모니터 ===================="
# 프로세스 / 어떤 setting 인지
PROC=$(pgrep -af "main.py --continuous" | grep -v pgrep | head -1)
if [ -n "$PROC" ]; then
  SET=$(echo "$PROC" | grep -oE -- "--setting [0-9]" | head -1)
  echo "상태       : 돌아가는 중  ($SET,  PID $(echo $PROC | awk '{print $1}'))"
else
  echo "상태       : 멈춤 (학습 프로세스 없음)"
fi

# 실시간 신호: runinfo 에피소드 reward (버퍼링 없음)
if ls logs/runinfo/*.monitor.csv >/dev/null 2>&1; then
  cat logs/runinfo/*.monitor.csv | grep -vE '^#|^r,l' | sort -t, -k3 -n > /tmp/_pctmon.csv
  tot=$(wc -l < /tmp/_pctmon.csv)
  early=$(head -50 /tmp/_pctmon.csv | awk -F, '{s+=$1;n++} END{if(n)printf "%.3f",s/n}')
  rec=$(tail -50 /tmp/_pctmon.csv | awk -F, '{s+=$1;n++} END{if(n)printf "%.3f",s/n}')
  reclen=$(tail -50 /tmp/_pctmon.csv | awk -F, '{s+=$2;n++} END{if(n)printf "%.0f",s/n}')
  upd=$(stat -c %y logs/runinfo/0.monitor.csv 2>/dev/null | cut -d. -f1)
  echo "에피소드   : $tot 개   (runinfo 갱신: ${upd##* })"
  echo "reward     : 초기50 $early  →  최근50 $rec      (≈적재율×10, 오르면 좋음)"
  echo "최근 박스수 : 평균 $reclen 개/에피소드"
fi

# 최신 학습 로그 자동 선택: 루트/logs 의 train_*.log + resume_*.log 중 가장 최근 수정된 것 = live.
LOG=$(ls -t logs/train_*.log train_*.log logs/resume_*.log 2>/dev/null | head -1)

# --- best reward (= 현재 run 의 역대 최고 mean_ratio). reward ≈ ratio × 10. ---
_fmt_best() {  # stdin: [best] 라인들 → "ratio R  (reward≈ R*10)"
  local r; r=$(grep -ohE "mean_ratio=[0-9.]+" | cut -d= -f2 | sort -g | tail -1)
  [ -n "$r" ] && awk -v r="$r" 'BEGIN{printf "ratio %.4f  (reward≈%.3f)", r, r*10}' || echo "(아직 기록 없음)"
}
CUR_BEST=$(grep -ah "\[best\]" "$LOG" 2>/dev/null | _fmt_best)
echo "-------------------------------------------------------"
echo "best(현재 run)  : $CUR_BEST   [$(basename "$LOG")]"
# 주의: 다른 run 의 best 와 직접 비교 금지 — 박스 분포가 다르면(cjrand=구 스펙, cjspec=대회 스펙) 무의미.

echo "-------------------------------------------------------"
echo "[$LOG 최신 — 버퍼링으로 지연될 수 있음]"
grep -aE "Updates|space ratio" "$LOG" 2>/dev/null | tail -4 || echo "  (아직 버퍼 안 비워짐 — 위 reward로 판단)"
echo "========================================================"
echo "팁: '최근50 reward'와 적재율이 한참 안 오르면 수렴 → pkill -f 'main.py --continuous'"
