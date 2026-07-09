# UOS-RoboStack — 물리 시뮬레이션 기반 랜덤 팔레타이징

> **과제명:** 물리 시뮬레이션 환경 기반의 랜덤 팔레타이징 알고리즘 (CJ Logistics 팔레타이징 경진대회)
>
> 본 저장소는 **PCT(Packing Configuration Tree) + GAT(Graph Attention) 강화학습 정책**을
> NVIDIA **Isaac Lab 물리 엔진**으로 포팅하여, 수학적 안정성 근사 대신 **실제 물리**로
> 적재 안정성을 검증·학습하고, 대회 규격의 **ONNX/CPU 제출 패키지**까지 생성하는 전체 파이프라인입니다.

---

## 목차
1. [핵심 개념 한눈에](#1-핵심-개념-한눈에)
2. [디렉토리 구조 (각 폴더 설명)](#2-디렉토리-구조-각-폴더-설명)
3. [모델 체크포인트 — 최신 .pt는 어디에?](#3-모델-체크포인트--최신-pt는-어디에)
4. [환경 설정 / 설치](#4-환경-설정--설치)
5. [코드 구성 (아키텍처)](#5-코드-구성-아키텍처)
6. [실행 방법](#6-실행-방법)
7. [성능 / 결과](#7-성능--결과)
8. [대회 규격 & 채점 방식](#8-대회-규격--채점-방식)
9. [주의사항 / FAQ](#9-주의사항--faq)

---

## 1. 핵심 개념 한눈에

- **문제:** 컨베이어로 무작위 순서·무작위 크기/무게의 박스가 도착하면, 1.2 × 1.0 × 1.25 m 팔레트에
  무너지지 않게 쌓아 **적재율(부피 채움 비율)** 을 최대화한다. (온라인, 버퍼 미사용 기준)
- **정책:** PCT로 현재 적재 상태에서 놓을 수 있는 후보 위치(EMS leaf, 최대 100개)를 생성하고,
  GAT 정책이 그중 하나(leaf index 0~99)를 고른다. 관측은 순수 PCT 그래프 `301 × 9 = 2709`.
- **두 개의 축:**
  | 축 | 쓰는 곳 | 물리 사용 |
  |---|---|---|
  | **학습 / 평가** | Isaac Lab 물리로 안정성 검증하며 정책 학습·채점 | ✅ Isaac |
  | **대회 제출** | ONNX 정책 + numpy 패커 (가볍게 추론만) | ❌ CPU only |
- **제출물:** `submit_terminal_t18_best_ae.zip` — `submit_baseline`의 안정화된 제출 코드를 기준으로,
  최신 `PCT-best.pt`를 ONNX로 변환해 내장한 최종 제출 ZIP.

---

## 2. 디렉토리 구조 (각 폴더 설명)

```
assignment2_project/  (= GitHub: yeonjae1009/UOS-RoboStack)
├── isaaclab_pallet/              ★ Isaac Lab 포팅 핵심 (학습/시각화/평가생성)
├── Online-3D-BPP-PCT/            ★ PCT/GAT 모델 라이브러리 (vendored) + ONNX 변환
├── palletizing_simulator/        ★ 공식 Isaac 물리 채점기 (정답지)
├── submit_baseline/              ★ 현재 제출 코드 기준선 (ONNX/CPU, Isaac 불필요, AE clamp 포함)
├── submit_terminal_t18_best_ae/  ★ 최신 terminal_ratio_t18 모델을 넣은 최종 제출 폴더
├── templete code/                ★ env가 읽는 config + packer (대회 제공 템플릿)
├── submit_terminal_t18_best_ae.zip      ← 최종 제출 ZIP
├── ISAAC_LAB_PORTING_GUIDE.md    포팅 가이드 문서
├── IMPROVEMENT_PLAN.md           개선 계획 문서
└── README.md                     (이 문서)
```

### 2.1 `isaaclab_pallet/` — Isaac Lab 포팅 핵심
실제 물리에서 학습/시각화/평가를 수행하는 본체.

| 파일 | 역할 |
|---|---|
| `pallet_packing_env.py` | **DirectRLEnv** — 박스를 packer가 정한 위치에 스폰 → PhysX로 안착 → drift/tilt/oob/height/collapse 판정. 박스 소스(랜덤/시퀀스), 보상, GPU 배치 메트릭 포함 |
| `pct_reward.py` | Isaac-free **numpy 보상/패킹 레이어** (원본 Online-3D-BPP 보상과 비트 동일하게 검증됨) |
| `packer_pool.py` | env별 packer를 직렬/병렬(워커 프로세스)로 굴리는 풀 (`observe→select→place→reward`) |
| `skrl_models.py`, `policies.py` | (초기 브링업용 대안 경로 — 메인 GAT 파이프라인엔 미사용) |
| `scripts/train_pallet_gat.py` | **학습 진입점** (GAT) |
| `scripts/play_pallet_gat.py` | **GUI/headless 롤아웃** (랜덤 또는 대회 시퀀스) |
| `scripts/eval_competition_generate.py` | `.pt` → 대회 결과 JSON 생성 (Isaac 불필요) |
| `scripts/run_overnight.sh` | 밤샘 학습 래퍼 (cjspec_v2 워밍스타트, 풀 시드 순환, 자동 재시작) |
| `scripts/watch_training.sh` | 학습 모니터 요약 |
| `scripts/test_equivalence.py` | 원본 PackingContinuous == 포팅 패커 비트동일 검증 |
| `scripts/test_packer_pool.py`, `test_pallet_env.py`, `benchmark_pallet_env.py` | 풀/환경 테스트·벤치마크 |
| `runs/reward_terminal18_finish15/` | **최신 reward sweep 체크포인트** — [3장](#3-모델-체크포인트--최신-pt는-어디에) 참고 |

### 2.2 `Online-3D-BPP-PCT/` — 모델 라이브러리 (vendored)
원본 [alexfrom0815/Online-3D-BPP-PCT](https://github.com/alexfrom0815/Online-3D-BPP-PCT)을 **대회용으로 커스터마이즈**하여 통째로 포함(vendor)한 폴더. **GAT 신경망 정의가 여기 있습니다.**

| 항목 | 역할 |
|---|---|
| `model.py` | **`DRL_GAT`** (실제 GAT 정책 신경망) |
| `tools.py`, `storage.py` | 관측 처리, 롤아웃 스토리지 |
| `pct_envs/PctContinuous0/` | 원본 수학 근사 환경 (`bin3D.py` = 안정성 근사 기준, 동일성 검증의 reference) |
| `givenData.py` | 대회 박스 스펙/`DENSITY_MAX` 상수 |
| `export_onnx_isaac.py` | **`.pt` → ONNX 변환** (strict-load 우회 + numpy 2.x 호환) |
| `export_to_contest.py` | `.pt` → 대회 JSON 변환 (참고용) |
| `logs/experiment/` | 학습 로그 + **baseline 체크포인트** (Git LFS) |

> 학습/시각화/ONNX변환은 모두 이 폴더의 `model`/`tools`/`storage`에 의존합니다.
> 단, **대회 제출 실행에는 불필요**합니다.

### 2.3 `palletizing_simulator/` — 공식 물리 채점기 (정답지)
참가자 결과(JSON)를 Isaac Sim에 드롭·안착시켜 **공식 점수**를 매기는 대회측 시뮬레이터.

| 파일 | 역할 |
|---|---|
| `simulator.py` | Isaac Sim으로 박스 안착 → 최종 위치 기록 |
| `evaluator.py` | 의도 vs 최종 비교 → drift/collapse/oob/height 판정 + 점수 공식 |
| `config/sim_config.yaml` | 팔레트 치수, 물리 재질, drift 임계(0.40m) 등 채점 규정 |
| `box_sequence/box_sequence_{0,1}.json` | **공식 입력 박스 시퀀스** (각 250개, JSONL) |
| `scene.py`, `buffer_manager.py`, `monitor.py` | 씬/버퍼/모니터 유틸 |

### 2.4 `submit_baseline/`, `submit_terminal_t18_best_ae/` — 대회 제출 패키지 소스
`python main.py`로 실행되는 **온라인 추론 코드**. ONNX(onnxruntime, CPU)만 사용 → Isaac 불필요.

| 파일 | 역할 |
|---|---|
| `main.py` | 실행 진입점 (box_sequence 읽고 → 배치 → 결과 JSON 저장) |
| `algorithm.py` | **`Palletizer`** (핵심): packer.observe → ONNX argmax → leaf 선택 → place. `submit_baseline`부터는 raw PCT 좌표와 제출 JSON 좌표를 분리하고, JSON 출력 좌표만 2mm 안쪽으로 clamp |
| `src/pct/packer.py` | EMS/안정성 패킹 기하 (순수 numpy) |
| `config/pct_config.yaml` | setting/holder/density 등 추론 설정 |
| `config/algorithm_config.yaml` | 팔레트 치수, 버퍼 크기 등 |
| `src/models/pct_model.onnx` | 학습된 GAT 정책 (ONNX) |

`submit_baseline`은 검증된 제출 코드의 기준선이다. 이후 제출 ZIP을 만들 때는 이 폴더를 복사하고,
`src/models/pct_model.onnx`만 새로 교체하는 방식으로 관리한다.

### 2.5 `templete code/` — env가 읽는 config + packer (대회 템플릿)
대회 제공 템플릿. **Isaac env(`pallet_packing_env.py`)가 런타임에 이 폴더의 `config/pct_config.yaml`과
`src/pct` 패커를 읽습니다.** (setting 3, `density_max=1597.02` 등 학습 기준값)

---

## 3. 모델 체크포인트 — 최신 .pt는 어디에?

> 체크포인트(.pt)는 **Git LFS**로 관리됩니다. `git lfs install` 후 `git clone`/`git lfs pull` 해야 실제 파일을 받습니다.

| 구분 | 경로 | 설명 |
|---|---|---|
| **★ 최신/최종 제출 사용** | `isaaclab_pallet/runs/reward_terminal18_finish15/terminal_ratio_t18_from_terminal_best/PCT-best.pt` | terminal_ratio_t18 reward로 학습한 최고 random eval 모델. log best **87.48** |
| 이어하기용 | `isaaclab_pallet/runs/reward_terminal18_finish15/terminal_ratio_t18_from_terminal_best/PCT-best-resume.pt` | best 모델의 weights+optimizer+update |
| 최신 스냅샷 | `isaaclab_pallet/runs/reward_terminal18_finish15/terminal_ratio_t18_from_terminal_best/PCT-latest.pt` | 해당 run의 마지막 모델 |
| 비교 run | `isaaclab_pallet/runs/reward_terminal18_finish15/finish_ratio_t15_from_finish_best/PCT-best.pt` | finish action 학습 포함. log best **86.04** |
| **Baseline (원본)** | `Online-3D-BPP-PCT/logs/experiment/cjspec_v2-2026.06.24-23-29-47/PCT-best.pt` | 수학 환경 학습 원본 (워밍스타트 출발점, 대회점수 92.3) |
| **제출용 ONNX** | `submit_terminal_t18_best_ae/src/models/pct_model.onnx` | 위 최신 `.pt`를 변환한 ONNX (제출 ZIP 내장) |

아키텍처 공통: `setting=3`, `internal_node_holder=200`, `leaf_node_holder=100`,
`embedding_size=64`, `normFactor=0.8`, `density_max=1597.01889805696`.

---

## 4. 환경 설정 / 설치

### 4.1 학습 / 평가용 (Isaac 필요)
- **하드웨어:** NVIDIA RTX GPU (개발 환경: RTX PRO 6000 Blackwell 96GB)
- **소프트웨어:** Isaac Sim 5.1, **Isaac Lab 0.54.4**, conda 환경 `env_isaaclab` (Python 3.11)
- 서버에서 user 계정으로 사용
- 가상환경으로 들어가서 실행해야함

```bash
conda activate env_isaaclab
python3 -c "import isaaclab; print(isaaclab.__version__)"   # 0.54.4 확인
```

### 4.2 대회 제출 실행용 (Isaac 불필요, 가벼움)
제출 패키지는 **numpy + onnxruntime + pyyaml** 만 있으면 됩니다 (`submit_baseline/requirements.txt`).

```bash
pip install numpy onnxruntime pyyaml
```

### 4.3 체크포인트 받기 (Git LFS)
```bash
git lfs install
git clone https://github.com/yeonjae1009/UOS-RoboStack.git
cd UOS-RoboStack
git lfs pull        # .pt 실제 파일 다운로드
```

---

## 5. 코드 구성 (아키텍처)

### 5.1 정책 (PCT + GAT)
1. **PCT (Packing Configuration Tree):** 현재 적재 상태에서 다음 박스를 놓을 수 있는
   안정적 후보 위치(EMS leaf)를 최대 100개 생성. (`src/pct/packer.py`, 순수 numpy)
2. **GAT 정책 (`DRL_GAT`):** 내부노드+잎노드+다음박스를 그래프로 임베딩 → 어텐션 →
   잎노드별 (마스킹된) 확률 → **argmax로 leaf index(0~99) 선택**.
3. **관측:** `[내부노드 200 | 잎노드 100 | 다음박스 1] × 9 = 301 × 9 = 2709`.
   setting 3에서는 다음박스 슬롯에 **정규화 밀도(=질량/부피/DENSITY_MAX)** 포함.

### 5.2 Isaac 학습 루프 (`train_pallet_gat.py`)
```
reset → (반복) 정책(obs)=leaf 선택 → env.step → packer가 위치 확정 → PhysX 안착
       → drift/tilt/oob/height/collapse 판정 → 보상 → A2C(actor+critic) 업데이트 → 체크포인트
```
- env는 제출 코드와 동일 계열의 PCT packer + 자체 `pct_reward`를 사용.
- 학습 스크립트는 `Online-3D-BPP-PCT`의 `model.DRL_GAT`/`tools`/`storage`를 사용.
- 실패 종료코드: `1`=invalid, `2`=drift, `3`=tilt, `4`=oob, `5`=완료, `6`=height초과, `7`=drop, `8`=collapse.

### 5.3 제출 추론 루프 (`submit_baseline/algorithm.py`)
```
for box in boxes:
    obs = packer.observe(size, density)         # numpy
    if 놓을 잎 없음: terminate
    probs = onnx_session.run(obs)                # CPU 추론
    leaf = leaf_region[argmax(probs)]
    packer.place(leaf)                           # 위치 확정
    결과기록(position[x,y,z 중심], rotation 0/90)
```
**물리는 안 돌립니다.** "물리 인지"는 그 ONNX 모델이 Isaac 물리에서 학습/검증됐다는 의미입니다.

중요: PCT 내부 좌표와 제출 JSON 좌표는 분리되어 있다.

- `_raw_sequence`: PCT 내부 검증/다음 후보 판단용. 원본 PCT 좌표를 그대로 유지한다.
- `sequence`: 제출 JSON용. 팔레트 경계 반올림 문제를 피하기 위해 `x/y`를 0.002m 안쪽으로 clamp한다.

이 분리 덕분에 PCT가 고른 박스 순서/회전/개수는 유지하면서, Isaac spawn 시 `-0.0005`, `1.0005`
같은 경계 초과 좌표로 채점 오류가 나는 문제를 피한다.

---

## 6. 실행 방법

> 아래 학습/시각화/평가 명령은 모두 `conda activate env_isaaclab` 후 프로젝트 루트에서 실행합니다.

### 6.1 학습 (GAT, Isaac 물리)
빠른 단발 학습:
```bash
python3 isaaclab_pallet/scripts/train_pallet_gat.py \
  --num-envs 32 --max-boxes 256 --updates 1000 \
  --learning-rate 1e-5 --save-interval 250 \
  --load-model Online-3D-BPP-PCT/logs/experiment/cjspec_v2-2026.06.24-23-29-47/PCT-best.pt \
  --run-name my_run --headless
```
밤샘 학습(자동 재시작 + 박스풀 시드 순환):
```bash
nohup bash isaaclab_pallet/scripts/run_overnight.sh > /tmp/overnight.out 2>&1 &
bash isaaclab_pallet/scripts/watch_training.sh          # 진행 모니터
```
주요 인자: `--num-envs` 환경 수, `--max-boxes` 박스 풀 크기, `--updates` 업데이트 수,
`--learning-rate`(1e-6=약한 미세조정, 1e-5=빠른 적응), `--load-model` 워밍스타트, `--resume` 재개.

현재 최고 모델은 아래 reward profile로 학습한 run에서 나왔다.

```bash
python3 isaaclab_pallet/scripts/train_pallet_gat.py \
  --reward-profile terminal_ratio_t18 \
  --num-envs 32 \
  --updates 2500 \
  --load-model isaaclab_pallet/runs/reward_terminal18_finish15/previous/reward_sweep_small_terminal_ratio/PCT-best.pt \
  --run-name terminal_ratio_t18_from_terminal_best \
  --headless
```

reward scale 공통값:

```text
floor_coverage        = 1.2
boundary_floor        = 0.8
corner_floor          = 0.6
height_smoothness     = 0.8
support               = 0.08
weak_support          = 0.08
elevation_penalty     = 0.2
```

실험별 차이:

| run | 핵심 차이 | random eval 점수 |
|---|---|---|
| `terminal_ratio_t18_from_terminal_best` | `terminal_ratio_reward=18`, finish action 없음 | log best **87.48** |
| `finish_ratio_t15_from_finish_best` | `terminal_ratio_reward=15`, finish action 있음 | log best **86.04** |

최종 제출은 점수가 더 높았던 `terminal_ratio_t18_from_terminal_best/PCT-best.pt`를 사용한다.

### 6.2 시각화 (GUI 롤아웃)
대회 시퀀스를 순서대로 투입해서 직접 보기:
```bash
python3 isaaclab_pallet/scripts/play_pallet_gat.py \
  --checkpoint isaaclab_pallet/runs/reward_terminal18_finish15/terminal_ratio_t18_from_terminal_best/PCT-best.pt \
  --box-source file \
  --box-sequence-path palletizing_simulator/box_sequence/box_sequence_0.json \
  --num-envs 1 --max-boxes 250 --steps 130 \
  --step-delay 0.3 --hold-seconds 30
```
- `--box-source random`(기본)이면 스펙 랜덤 박스. `--box-source file`이면 대회 시퀀스(셔플 없음).
- 여러 팔레트 동시: `--num-envs 9`. 결정적 재현: `--device cpu`.

### 6.3 평가 (공식 점수 산출)
**1단계 — 정책으로 배치 JSON 생성 (Isaac 불필요):**
```bash
python3 isaaclab_pallet/scripts/eval_competition_generate.py \
  --checkpoint isaaclab_pallet/runs/reward_terminal18_finish15/terminal_ratio_t18_from_terminal_best/PCT-best.pt \
  --out-dir /tmp/pct_results
```
**2단계 — 공식 simulator로 물리 채점:**
```bash
# sim_config.yaml의 experience 경로가 이 머신과 다르면 먼저 교정 필요:
#   /home/user/isaacsim/apps/isaacsim.exp.base.python.kit
python3 palletizing_simulator/simulator.py \
  --config palletizing_simulator/config/sim_config.yaml \
  --input-dir /tmp/pct_results -o /tmp/sim_out
# 결과: /tmp/sim_out/result.json (episode별 점수 + 평균)
```

### 6.4 ONNX 변환 (.pt → .onnx)
```bash
cd Online-3D-BPP-PCT
python3 ../tools/export_pct_no_gym.py \
  --model-path ../isaaclab_pallet/runs/reward_terminal18_finish15/terminal_ratio_t18_from_terminal_best/PCT-best.pt \
  --out /tmp/pct_terminal_t18_best.onnx \
  --setting 3 --internal-node-holder 200 --leaf-node-holder 100
# torch vs onnxruntime 출력 차이와 argmax를 검증
```

### 6.5 제출 패키지 실행 / 빌드
**제출 코드 직접 실행 (채점관과 동일):**
```bash
cd submit_terminal_t18_best_ae
cp -r ../palletizing_simulator/box_sequence ./box_sequence   # 대회가 제공하는 입력
python3 main.py        # → algorithm_results/*.json 자동 생성
```
**제출 ZIP:** `submit_terminal_t18_best_ae.zip` (이미 빌드됨).
ZIP 최상위에 `main.py`가 있고, 압축 해제 후 `python main.py`로 실행되면 결과가 자동 산출됩니다.

---

## 7. 성능 / 결과

로컬 `main.py` 샘플 실행 결과(`assignment2/box_sequence` 기준):

| 제출 폴더 | seq_0 output | seq_1 output | total output | 비고 |
|---|---|---|---|---|
| `submit_baseline` | 96 | 72 | 168 | AE clamp 적용 기준선 |
| **`submit_terminal_t18_best_ae`** | 96 | 73 | 169 | 최신 `terminal_ratio_t18` ONNX 적용 |

- `submit_terminal_t18_best_ae`는 경계 초과 AABB 검사에서 `bounds_bad=0`.
- 변환 검증: `PCT-best.pt → pct_model.onnx` export 후 dummy 입력에서 `max_abs_diff=1.1324883e-06`,
  `argmax_torch == argmax_onnx`.
- 학습 로그 기준 random eval best는 `terminal_ratio_t18_from_terminal_best`가 **87.48**로 가장 높았다.

---

## 8. 대회 규격 & 채점 방식

- **팔레트:** 1.2(L) × 1.0(W) × 1.25(H) m, 두께 0.15 m.
- **박스:** W,L ∈ [0.17, 0.32] m, H ∈ [0.13, 0.26] m, 질량 ∈ [0.5, 6.0] kg (연속 랜덤). 회전 0° / Z축 90°.
- **적재율:** `∑(성공 박스 부피) / (1.2·1.0·1.25)`.
- **실패 = 0점(치명적):** 박스 성공률 100% 미만(드리프트>0.40m로 collapse, oob, drop 중 하나라도) 또는
  적재 높이 초과 시 그 에피소드 **0점**.
- **점수:** `적재율×100 + (20 - 버퍼사용수)`, 최대 120점. 데이터셋당 처리 ≤ 90초.
- **제출:** 소스 전체 ZIP 1개. 최상위 `main.py`, 설정은 `config/` YAML. `python main.py`로 실행되어 결과 자동 산출.

---

## 9. 주의사항 / FAQ

- **체크포인트가 안 보여요:** Git LFS 미설치/미pull. `git lfs install && git lfs pull`.
- **GUI가 안 떠요 / 멈춰요:** Isaac GUI는 본인 터미널에서 실행하세요 (백그라운드 무TTY는 멈춤).
  현재 최종 체크포인트는 `terminal_ratio_t18_from_terminal_best/PCT-best.pt`입니다.
- **`simulator.py` 실행 실패:** `config/sim_config.yaml`의 `app.experience` 경로가 머신과 다를 수 있습니다.
  설치된 kit 경로(예: `/home/user/isaacsim/apps/isaacsim.exp.base.python.kit`)로 교정하세요.
- **제출물은 Isaac을 쓰지 않습니다:** ZIP 안은 numpy+onnxruntime 패커뿐입니다. "Isaac 기반"은
  내부 ONNX 모델이 Isaac 물리에서 학습/검증되었음을 뜻합니다.
- **reward scale은 제출 ZIP에 직접 들어가지 않습니다:** reward scale은 학습 중 `PCT-best.pt` 가중치를
  만드는 데만 쓰입니다. 제출 시에는 해당 `.pt`를 ONNX로 변환한 `src/models/pct_model.onnx`만 실행됩니다.
  따라서 새 reward 실험 결과를 제출에 반영하려면 `.pt → ONNX` 변환 후 `submit_baseline`의 ONNX를 교체합니다.
- **새 제출 코드를 만들 때 기준 폴더:** `submit_baseline`을 복사한 뒤 `src/models/pct_model.onnx`만 교체합니다.
  `algorithm.py`의 raw PCT 좌표/JSON clamp 분리 로직은 그대로 유지해야 합니다.
- **GPU 비결정성:** play 스크립트는 GPU에서 run마다 미세하게 다를 수 있습니다. 제출/채점 경로(CPU/ONNX)는 결정적입니다.

---

*GitHub: [yeonjae1009/UOS-RoboStack](https://github.com/yeonjae1009/UOS-RoboStack)*
