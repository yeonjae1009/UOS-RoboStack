# PCT → Isaac Lab 포팅 가이드

> 목적: 현재의 PCT 팔레타이징 환경을 **Isaac Lab 병렬 물리 RL**로 옮기기.
> 결정 사항: ① 목표 = 강건성 + 밀도, **파이프라인 구축 자체가 목표** / ② 학습기 = **Isaac Lab 표준 RL** 사용.
> 작성 기준 시스템에서 확인: **Isaac Lab 2.3.2**, RL 라이브러리 `rl_games / rsl_rl / skrl / sb3` 사용 가능.

---

## 0. 핵심 전제 (틀린 전제 위에 설계하지 말 것)

1. **"가변 액션" 벽은 이미 작다.** PCT는 이미 고정 패딩 + 마스킹 구조다.
   - 관찰 = `(internal 200 + leaf 100 + next 1) × 9` = **고정 2709차원**
   - 액션 = leaf index 0~99 = **고정 `Discrete(100)` + valid 마스크**
   - 즉 leaf 개수가 변해도 100으로 패딩하고 무효 leaf를 마스킹 → 표준 invalid-action masking으로 처리 가능.

2. **진짜 벽은 CPU 패킹 기하다.** EMS/leaf 후보 생성 + convex-hull 안정성 계산(`space.py`, 순수 numpy, env당 순차). **Isaac의 GPU 물리 가속은 이 부분을 못 줄인다.** 수천 env를 띄워도 leaf 생성이 CPU 직렬이면 GPU가 논다. → 이게 포팅의 실질 난관이며 GPU가 아니라 코드 설계로 푼다.

3. **물리-in-the-loop는 "강건성"을 주지 "밀도"를 자동으로 안 준다.** 현재 모델은 이미 collapse/oob/drop=0 (안정성 해결됨). "빈 공간(점유율 72.3%)"은 패킹 밀도 문제이고, **reward 설계로 따로 유도**해야 한다. 포팅의 자동 보너스가 아니다.

---

## 1. 다른 PC로 가져갈 자산 (zero에서 시작 아님)

| 가져갈 파일 (이 레포 기준 경로) | Isaac Lab에서의 역할 |
|---|---|
| `Online-3D-BPP-PCT/templete code/src/pct/packer.py` | **CPU 패킹 코어 드라이버** (torch 없는 순수 numpy). env당 leaf 후보 생성·배치·점유율 = 단계 ①⑤. 이미 추출돼 있어 그대로 씀. |
| `.../src/pct/space.py`, `convex_hull.py`, `PctTools.py` | packer가 의존하는 기하/안정성 로직 (순수 numpy) |
| `palletizing_simulator/simulator.py`, `scene.py` | **박스 스폰 → 낙하 → 안착 → pose 읽기 → 지지/드리프트 평가** 가 이미 Isaac Sim으로 구현됨 = 단계 ③④의 원본. **Stage 1~2는 이걸 Isaac Lab DirectRLEnv로 이식하는 작업.** |
| `palletizing_simulator/evaluator.py` | 점수/성공 판정 로직 (reward 설계 참고) |
| `Online-3D-BPP-PCT/model.py`, `attention_model.py`, `graph_encoder.py` | GAT 정책 (재사용 시) |
| `Online-3D-BPP-PCT/givenData.py` | 대회 박스 스펙 + `DENSITY_MAX=1597.0` |

> 박스 스펙(참고): W,L ∈ [0.17, 0.32] m, H ∈ [0.13, 0.26] m, 질량 ∈ [0.5, 6.0] kg.
> density 입력 = `mass / 부피 / DENSITY_MAX` (DENSITY_MAX = 6.0 / (0.17·0.17·0.13) ≈ 1597.0). setting 3.

---

## 2. 환경 셋업 (첫 관문)

- Isaac Sim + **Isaac Lab 버전을 맞출 것 (2.3.2)**. 버전 불일치가 가장 흔한 삽질 포인트.
- 검증: Isaac Lab 기본 예제(예: cartpole direct)를 GPU 병렬로 띄워 환경이 정상 동작하는지 먼저 확인.
- `DirectRLEnv` 템플릿 위치: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/`
  - 구조 단순: `cartpole`
  - 물체 조작·pose 읽기 예: `franka_cabinet`, `factory`

---

## 3. 단계별 진행 (★ Stage 1이 make-or-break)

### Stage 1 ★ — 단일 env "낙하 + pose 회수" 증명 (RL 없음)
- `DirectRLEnv` 스켈레톤(`num_envs=1`)에 팔레트 + 박스 1개 스폰.
- `packer.py`로 leaf 후보 1개 계산 → **leaf(팔레트 로컬) → 월드좌표 변환** → 박스를 그 위에 놓고 PhysX로 안착.
- 안착 후 pose 읽어서 **"의도 위치 vs 실제 위치(drift)"** 출력.
- 이게 되면 "가변 액션 + 물리 결합"의 핵심이 풀린 것. **안 되면 여기서 멈추고 이것부터 해결.**
- 참고: `palletizing_simulator/scene.py`의 spawn/settle/readback 로직을 Isaac Lab scene API로 이식.

### Stage 2 — DirectRLEnv 완성 (작은 num_envs)
- `_get_observations`: packer로 `(301×9)` 관찰 + **leaf 마스크** 생성 (CPU).
- action: `Discrete(100)` (leaf index).
- `_apply_action`: 선택 leaf → 박스 낙하 (Stage 1 로직).
- `_get_rewards`: 부피 보상 + **실물리 드리프트/붕괴** 페널티.
- `_get_dones`: 더 놓을 leaf 없음 / 붕괴 / 높이 초과 시 종료.
- 1 롤아웃 돌려서 shape·흐름만 확인 (학습 아직 아님).

### Stage 3 — num_envs 확장 (16 → 256 → 1024 …) + 병목 대응
- 측정: ①⑤(CPU packer)가 직렬이면 처리량이 무너진다.
- 대응 선택지:
  - (a) packer를 **멀티프로세스/스레드로 분산** (env 그룹별 병렬)
  - (b) EMS leaf 생성을 **heightmap 기반 GPU-배치형으로 단순화** (바닐라 PCT에서 일부 이탈, 대신 진짜 병렬)
- ★ 여기서 96GB GPU가 빛난다 — **물리(낙하·안착)는 GPU 배치라 거의 공짜로 빨라짐**. 병목은 오직 leaf 생성.

### Stage 4 — 표준 RL 라이브러리 연결 (학습)
아래 4절 참고.

---

## 4. Stage 4 — RL 라이브러리 & 마스킹 (가장 중요한 함의)

**확인된 사실: Isaac Lab 2.3.2의 RL 래퍼엔 action masking 기본 배선이 없다** (`isaaclab_rl`에 action_mask 흔적 없음).

> **그래서 마스킹은 "정책 네트워크 안에서" 처리한다.**
> env가 관찰에 leaf 마스크를 실어 보내고, 정책이 무효 leaf의 logit을 −∞로 만든 뒤 categorical 분포.
> **이것이 PCT를 표준 RL에 끼우는 핵심 트릭.**

| 라이브러리 | 추천도 | 이유 |
|---|---|---|
| **skrl** | ★ 추천 | 커스텀 모델(GAT) + 관찰 기반 마스킹을 가장 깔끔하게 정의. Isaac Lab 통합. |
| rl_games | 가능 | 가장 "표준"이나 GAT custom network builder + 마스킹 배선이 번거로움 |
| rsl_rl | 비추 | MLP 중심, 마스킹/그래프넷 끼우기 어려움 |

**현실적 순서 (파이프라인 우선)**:
1. Stage 4 첫 시도는 **GAT 말고 단순 MLP/transformer-lite 정책 + 마스킹**으로 end-to-end "돈다"를 확인.
2. env가 완성된 뒤 정책만 **GAT로 교체** (이 시점엔 교체가 쉬움).

---

## 5. DirectRLEnv 구현 시 채울 메서드 (2.3.2 기준)

```
_setup_scene       : 팔레트 N개 + 박스 풀 스폰
_pre_physics_step  : action(leaf idx) 받아 저장
_apply_action      : 선택 leaf → 박스 낙하 트리거
_get_observations  : packer로 (301×9) + leaf_mask  ← CPU 병목 지점
_get_rewards       : 부피 + 실물리 드리프트/붕괴
_get_dones         : leaf 없음 / 붕괴 / 높이초과
_reset_idx         : 해당 env 팔레트 비우고 박스 시퀀스 재생성
```

**안착(settle) 처리**: env.step()의 물리 substep을 고정 횟수로 돌리거나(현재 `palletizing_simulator`의 `settling.max_steps/final_steps` 방식 재활용) 속도 임계값으로 안착 판정. 배치 env에선 **고정 substep**이 단순·안전.

---

## 6. 리스크 체크리스트 (가면서 기억)

- [ ] **Stage 1이 전부다.** leaf→월드→낙하→pose 결합부가 거기서 풀린다. 안 되면 GPU 무관하게 진도 0.
- [ ] **Stage 3의 CPU leaf 병목**이 두 번째 벽. packer 병렬화/단순화가 실제 처리량을 가른다.
- [ ] **마스킹은 정책 내부에서.** 표준 RL 래퍼에 기댈 수 없음.
- [ ] **밀도(빈 공간)는 reward로 따로 유도.** 물리 학습의 자동 보너스 아님.
- [ ] **버전 정합(Isaac Lab 2.3.2).** 불일치 시 import/scene API가 깨짐.

---

## 7. 첫 번째 할 일 (다른 PC 도착 후)

1. Isaac Lab 2.3.2 정상 동작 확인 (예제 1개 실행).
2. `direct/cartpole`를 복사해 `direct/pallet_packing` 스켈레톤 생성.
3. **Stage 1** 구현: 팔레트 1개 + 박스 1개, `packer.observe()`로 leaf 1개 얻어 월드좌표로 낙하 → pose 읽기.
   - 이때 `palletizing_simulator/scene.py`의 박스 스폰·안착·pose 회수 코드를 옆에 띄워놓고 그대로 이식.
4. drift(의도 vs 실제)가 작게 나오면 → Stage 2로.

---

## 참고: 현재 베이스라인 (포팅 전 도달점)
- 학습: 빠른 근사 env(현재 PCT) + Isaac 평가 루프 → **Isaac 물리 채점 92.3점** (183박스 100% 성공, collapse/oob/drop 0).
- 제출물: `~/submission_pct_cjspec_v2.zip` (torch 없이 onnxruntime 추론).
- 즉 **포팅이 실패해도 92.3 베이스라인은 확보됨.** 포팅은 그 위에서 강건성·밀도를 더 짜내는 연구 트랙.
