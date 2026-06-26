# CJ 팔레타이징 — PCT(DRL) 작업 가이드

이 문서 하나로 **학습 → 평가 → Isaac 검증 → ONNX 교체 → 제출 ZIP** 전 과정을 진행할 수 있다.
(작성 기준: 2026-06-23, 초기 1시간 체크포인트로 Isaac 채점 **평균 87.80점** 달성한 상태)

---

## 0. 큰 그림 — 두 개의 분리된 세계

```
[학습 공장]  Online-3D-BPP-PCT/          ← torch로 PCT 모델 학습 (제출 ❌)
   .venv-pct (Python3.9 + torch1.10+cu113 + gym0.13)
        │  학습된 정책을 ONNX로 export
        ▼
[제출물]     templete code/              ← algorithm.py가 ONNX 추론 (제출 ✅)
   .venv (Python3.12 + requirements.txt: numpy/onnxruntime, torch 없음)
        │  python main.py → 적재계획 JSON
        ▼
[채점기]     palletizing_simulator/       ← Isaac Sim 물리 검증/점수 (대회측 환경)
   /home/robotics/isaac-sim-5.1/bin/python
```

- **학습은 torch**(`.venv-pct`), **제출/추론은 ONNX+numpy**(`.venv`). 절대 섞지 말 것.
- 제출하는 건 `templete code/` 내용물뿐. PCT repo·simulator는 제출 안 함.

---

## 1. 경로 / 환경 빠른 참조

| 용도 | 경로 |
|---|---|
| PCT 학습 repo | `~/Documents/assignment2/Online-3D-BPP-PCT` |
| 학습용 venv (torch) | `.venv-pct/bin/python` (위 repo 안) |
| 제출물 (대회 코드) | `~/Documents/assignment2/templete code` |
| 제출 환경 venv (onnx) | `~/Documents/assignment2/.venv/bin/python` |
| Isaac 시뮬레이터 | `~/Documents/assignment2/palletizing_simulator` |
| Isaac Python | `/home/robotics/isaac-sim-5.1/bin/python` |
| 완성 제출 ZIP | `~/submission_pct.zip` |

> 모든 명령은 해당 폴더로 `cd` 한 뒤 실행. 경로에 공백(`templete code`)이 있으면 따옴표로 감쌀 것.

---

## 2. 학습 (PCT repo, torch)

### 2-1. 새로 학습 시작
```bash
cd ~/Documents/assignment2/Online-3D-BPP-PCT
bash train_cj.sh cj_v1          # 백그라운드 아님 — 터미널 점유. 끄려면 Ctrl+C
# 백그라운드로:
echo cj_v1 | nohup .venv-pct/bin/python main.py --continuous --setting 1 \
  --internal-node-holder 200 --leaf-node-holder 100 --num-processes 16 \
  > train_cj_v1.log 2>&1 &
```

### 2-2. 이어서 학습 (warm start — 체크포인트에서 계속)
```bash
CKPT=$(ls -t logs/experiment/cj_v1*/PCT-*.pt | head -1)
echo cj_v2 | nohup .venv-pct/bin/python main.py --continuous --setting 1 \
  --internal-node-holder 200 --leaf-node-holder 100 --num-processes 16 \
  --load-model --model-path "$CKPT" > train_cj_v2.log 2>&1 &
```

### 2-3. 진행 상황 보기
```bash
bash monitor.sh                 # 에피소드 수 / 평균 reward(≈적재율×10) / 최신 적재율
```

### 2-4. 언제 멈추나
- `bash monitor.sh`의 **"mean space ratio / 최근 reward"가 한참(예: 1시간) 안 오르면 수렴** → 멈춰도 됨.
- 멈추기: `pkill -f 'main.py --continuous'`
- 체크포인트는 `logs/experiment/<이름>-<날짜>/PCT-*.pt`에 200 updates마다 자동 저장(멈춰도 보존).

> 주의: `--num-processes`는 CPU 코어 수에 맞춰(이 PC 16). 너무 크면 느려짐.
> 주의: `--internal-node-holder`(200)는 한 에피소드 최대 박스 수 상한. 박스 더 쌓는 모델이면 키울 것(메모리/속도 ↑).

---

## 3. 평가 — 기하 적재율 빠른 확인 (Isaac 없이)

```bash
cd ~/Documents/assignment2/Online-3D-BPP-PCT
CKPT=$(ls -t logs/experiment/cj_v1*/PCT-*.pt | head -1)
echo eval | .venv-pct/bin/python evaluation.py --evaluate --load-model \
  --model-path "$CKPT" --load-dataset --dataset-path dataset/setting123_discrete.pt \
  --setting 1 --internal-node-holder 200 --leaf-node-holder 100 --evaluation-episodes 10
```
> 이건 PCT 자체 환경의 기하 적재율(물리 X). 빠르게 "모델이 나아졌나" 보는 용도.

---

## 4. Isaac Sim으로 결과 보기 / 물리 점수

### 4-1. 최신 체크포인트로 대회 시퀀스 적재 → Isaac 입력 폴더로 복사
```bash
cd ~/Documents/assignment2/Online-3D-BPP-PCT
bash export_and_view.sh         # 두 시퀀스 변환 + palletizing_simulator/algorithm_results/로 복사
```

### 4-2. 점수만 (헤드리스, 빠름)
```bash
cd ~/Documents/assignment2/palletizing_simulator
/home/robotics/isaac-sim-5.1/bin/python simulator.py
# → sim_results/result.json (점수) + *_result.png (스크린샷)
```

### 4-3. GUI로 적재 과정 보기
```bash
cd ~/Documents/assignment2/palletizing_simulator
bash run_gui.sh                 # 창이 뜨면 ▶ PLAY 클릭
```
> GUI는 CPU를 많이 씀 — 학습 중이면 잠깐 멈추고(`pkill ...`) 보는 게 매끄럽다.
> 점수는 `sim_results/result.json`에 저장됨 (stdout은 Isaac 내부 로그로 가서 잘 안 보임).

---

## 5. ★ 제출물 모델 교체 (학습 더 한 뒤 핵심 과정)

학습으로 모델이 좋아졌으면, **ONNX만 새로 export → 제출물 교체 → rezip** 하면 끝.

```bash
cd ~/Documents/assignment2/Online-3D-BPP-PCT

# (1) 최신 체크포인트 → ONNX (torch와 일치 검증까지 자동)
CKPT=$(ls -t logs/experiment/cj_v1*/PCT-*.pt | head -1)
.venv-pct/bin/python export_onnx.py --model-path "$CKPT" --out pct_model.onnx \
  --internal-node-holder 200 --leaf-node-holder 100
#   → "argmax 일치: N/N (OK)" 나오면 성공

# (2) 제출물의 모델 교체
cp pct_model.onnx "../templete code/models/pct_model.onnx"

# (3) 클린 환경에서 동작 확인 (torch 없는 .venv)
cd "../templete code"
/home/robotics/Documents/assignment2/.venv/bin/python main.py   # 적재율 확인

# (4) (선택) Isaac 물리 점수 재확인
cp algorithm_results/*.json ../palletizing_simulator/algorithm_results/
cd ../palletizing_simulator && /home/robotics/isaac-sim-5.1/bin/python simulator.py
#   → sim_results/result.json

# (5) 제출 ZIP 재생성  (6장 참고)
```

> ⚠️ holder 값(200/100)은 **학습 = export_onnx = pct_config.yaml** 세 곳이 항상 같아야 함. 학습 때 바꿨으면 export·config도 같이 바꿀 것.

---

## 6. 제출 ZIP 만들기 + 검증

```bash
cd "/home/robotics/Documents/assignment2/templete code"
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null

rm -f ~/submission_pct.zip
zip -r ~/submission_pct.zip \
    main.py algorithm.py buffer_manager.py visualize.py \
    config src models requirements.txt README.md \
    -x "*/__pycache__/*" "*.pyc" "*.orig"

unzip -l ~/submission_pct.zip          # main.py가 최상위에 있는지 확인
```

### 제출 전 최종 검증 (채점 서버 흉내)
```bash
rm -rf /tmp/zipcheck && mkdir /tmp/zipcheck && cd /tmp/zipcheck
unzip -q ~/submission_pct.zip
cp -r "/home/robotics/Documents/assignment2/templete code/box_sequence" .   # 채점기가 입력 제공하는 상황
/home/robotics/Documents/assignment2/.venv/bin/python main.py               # 종료코드 0 + 결과 생성이면 OK
```

### 제출 규정 체크리스트
- [ ] `main.py`가 ZIP **최상위** (`unzip -l`로 확인)
- [ ] `python main.py` 종료코드 0, `algorithm_results/*.json` 생성
- [ ] torch/gym/scipy import 없음 (numpy + onnxruntime만)
- [ ] AI 모델은 `models/pct_model.onnx` (ONNX 형식)
- [ ] 제공 파일(main/buffer_manager/visualize) 무수정
- [ ] 모든 설정값이 `config/` YAML
- [ ] `config/algorithm_config.yaml`의 `buffer.size: 0`
- [ ] 데이터셋당 처리시간 < 90초 (현재 ~5초)

> ZIP 파일명은 자유(`mv`로 변경 OK). **푼 뒤 내부 구조만 유지**되면 됨.

---

## 7. 내가 만든 파일들 (역할)

### PCT repo (`Online-3D-BPP-PCT/`)
| 파일 | 역할 |
|---|---|
| `train_cj.sh` | 대회 스펙 학습 런처 (continuous/setting1/holder200·100/proc16) |
| `monitor.sh` | 학습 진행 모니터 |
| `export_onnx.py` | GAT 정책 → ONNX (+ torch 일치 검증) |
| `export_to_contest.py` | 체크포인트로 대회 시퀀스 적재 → 대회 JSON (오프라인) |
| `export_and_view.sh` | 두 시퀀스 export + simulator 폴더로 복사 |
| `RUN_COMMANDS.txt` | 명령 모음 |
| `givenData.py` | **대회 박스 스펙** (5종 + 팔레트 1.2×1.0×1.25) — 수정함 |
| `pct_envs/PctContinuous0/space.py` | 용량 초과 시 정상종료 가드 추가(패치) |

### 제출물 (`templete code/`)
| 파일 | 역할 |
|---|---|
| `algorithm.py` | **PCT 추론** (Packer로 관찰값 → onnxruntime → 배치) |
| `config/pct_config.yaml` | 모델경로/holder/size_minimum/setting |
| `config/algorithm_config.yaml` | 팔레트/회전/**buffer.size: 0** |
| `src/pct/space.py,convex_hull.py,PctTools.py` | 패킹 기하 (순수 numpy 포팅) |
| `src/pct/packer.py` | bin3D의 torch 없는 온라인 드라이버 |
| `models/pct_model.onnx` | 학습된 정책 (ONNX) |

---

## 8. 핵심 파라미터 — 어디서 바꾸나

| 파라미터 | 위치 | 비고 |
|---|---|---|
| 박스 5종 / 팔레트 크기 | `givenData.py` (학습), `templete code`의 PalletConfig는 main.py가 config에서 | 대회 데이터에서 추출한 값 |
| 회전 0/90 | `--setting 1` (자동) | setting1=rot{0, Z90} = 대회 제약과 일치 |
| 버퍼 0 | `config/algorithm_config.yaml` buffer.size | 보너스 +20 (천장) |
| holder (200/100) | train 명령 / export_onnx / pct_config.yaml | **세 곳 동일 필수** |
| normFactor (0.8) | ONNX에 baked-in (1/max(container)) | 컨테이너 바꾸면 재export |
| size_minimum (0.134) | pct_config.yaml | 5종 최소 치수 |

---

## 9. 트러블슈팅 / 알아둘 것

- **IndexError (학습 중)**: 박스가 holder 초과. holder 키우거나, space.py 가드(이미 패치)로 정상종료됨.
- **numpy 2.x 에러**: 제출 환경은 numpy 2.4.3. `np.float` 등 제거됨 → `np.float64` 사용(이미 수정). 새 코드 포팅 시 주의.
- **ONNX opset 에러**: torch 1.10은 opset ≤ 15. `export_onnx.py`는 opset 13 사용.
- **GUI 크래시**: `run_gui.sh`가 DISPLAY/XAUTHORITY 강제 + ROS 환경 제거 + 재시도. 그래도 죽으면 헤드리스(`simulator.py`)로.
- **물리 간극**: PCT는 기하 적재율만 최적화(물리 모름). 현재는 EMS 적층이 support를 자연 충족해 통과(collapse 0). **모델/시퀀스 바뀌면 Isaac 재검증 필수.**
- **점수 = 적재율(%) + (20 − buffer_size)**. buffer 0 → 파일당 +20. 현재 87.80에 이미 포함.

---

## 10. 다음 개선 아이디어

1. **학습 더** (제일 쉬움): 수렴까지 돌리면 적재율↑ → 점수↑. ONNX만 교체.
2. **holder 튜닝**: 박스를 더 쌓게 holder↑ (속도 trade-off).
3. **보상에 물리 반영**: 현재 보상은 부피만. support/안정성 패널티를 env에 추가하면 물리 견고성↑ (난이도 높음).
4. **leaf_node_holder 조정**: 후보 수 ↓ → 속도↑(시간 빠듯할 때), ↑ → 선택지↑(점수 여지).
5. **여러 체크포인트 Isaac 비교** → 실제 점수 최고인 모델 선택 (기하 적재율 ≠ 물리 점수일 수 있음).

---

## 11. 현재 상태 요약 (스냅샷)

- 제출 ZIP: `~/submission_pct.zip` — **그대로 제출 가능, Isaac 87.80점 검증됨**
- 모델: 1시간 학습 체크포인트 (더 학습 시 상승 여지)
- baseline 16.35 → PCT **87.80** (성공률 100%, 무너짐 0)
- 처리시간: 데이터셋당 ~5초 (90초 제한 대비 18배 여유)
