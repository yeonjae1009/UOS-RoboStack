# Isaac 머신 검증 체크리스트 (isaaclab_pallet)

이 머신(Isaac 없음)에서 이미 **수치로 통과한 것**과, **Isaac 머신에서만 검증 가능한 것**을 구분한다.
목표 배포 형태: **학습된 GAT(.pt) → ONNX → `submission_UOS-robostack/algorithm.py` 추론**
(입력 = packer가 만든 PCT 그래프 관찰, 출력 = leaf 인덱스, 위치는 packer가 디코딩).

---

## 이미 통과 (이 머신, Isaac 불필요) — 재현용
- [ ] CPU layer 동일성: `python3 isaaclab_pallet/scripts/test_equivalence.py --max-boxes 250 --policy cycle`
      → 원본 PCT vs 이식 reward/metric **0.000e+00** (bit-exact) 확인.
- [ ] 병렬 packer 정확성/속도: `python3 isaaclab_pallet/scripts/test_packer_pool.py --num-envs 16 --workers 4`
      → 직렬 vs 병렬 **0 불일치**, 속도 향상 확인.

---

## Stage 0 — 환경 정합
- [ ] Isaac Lab **2.3.2** + Isaac Sim 버전 일치 확인.
- [ ] 표준 예제 1개 GPU 병렬 실행 (예: `direct/cartpole`)으로 설치 sanity.
- [ ] `templete code/src/pct`, `Online-3D-BPP-PCT` 가 import 경로에 있는지(이식 env가 packer를 거기서 로드).

## Stage 1 — 단일 env 스모크 (물리 결합의 핵심)
- [ ] `python3 isaaclab_pallet/scripts/test_pallet_env.py --num-envs 1 --max-boxes 8 --policy first`
- [ ] **spawn-z 수정(④) 확인**: 박스가 packer 안착 높이에 스폰돼야 함. `terminal_drift`가 작게(예: < 0.05 m) 나오는지.
      (수정 전엔 EMS leaf z로 스폰돼 최대 0.64 m 낙하 → drift/drop 오발동.)
- [ ] **done_reason 코드 동작**: 1=무효/leaf없음, 2=drift, 3=tilt, 4=oob, 5=완료, 6=높이, 7=drop, **8=스택붕괴**.
- [ ] **settle(A1) 확인**: `settle_max_steps`>0일 때 안착 후 속도 수렴. 너무 길면 느리고 너무 짧으면 drift 과측정 → `settle_vel_threshold` 튜닝.
- [ ] 시각화로 박스가 실제로 팔레트 위에 안정적으로 쌓이는지 육안 확인.

## Stage 2 — 누적 안정성(A2) 임계값 튜닝
- [ ] 일부러 불안정하게 쌓아 `stack_drift_fail_threshold`(기본 0.12 m)가 **이전 박스 붕괴**를 done_reason 8로 잡는지.
- [ ] 정상 적재인데 8이 오발동하면 임계값 상향. 실제 붕괴를 놓치면 하향.
- [ ] `extras["last_stack_drift"]`/`terminal_stack_drift` 로깅으로 분포 확인.

## Stage 3 — 병렬 packer 처리량 (CPU 병목 = 가이드의 "진짜 벽")
- [ ] num_envs 스윕: 16 → 64 → 256 → 1024.
- [ ] `cfg.num_packer_workers` 0(직렬) vs N(=물리 코어 수)로 **실 env step FPS** 비교.
      (CPU 정확성은 이미 bit-identical 증명됨. 여기선 물리+풀 결합 처리량만 측정.)
- [ ] GPU util 확인: leaf 생성이 병목이면 GPU가 논다 → workers 증가 효과 확인.
- [ ] spawn 워커 IPC 오버헤드 vs 이득 손익분기점 찾기(작은 num_envs는 직렬이 빠를 수 있음).

## Stage 4 — 학습
- [ ] **(파이프라인 우선)** skrl MLP 먼저 end-to-end 돌아가는지:
      `python3 isaaclab_pallet/scripts/train_pallet_skrl.py --num-envs 64 --headless`
      → 마스킹 정책이 유효 leaf만 고르고 reward가 오르는지. (이건 **배포용 아님**, 배선 확인용.)
- [ ] **(배포용)** GAT 학습: `python3 isaaclab_pallet/scripts/train_pallet_gat.py ...`
      → `extras["pct_obs"]`(순수 2709) 사용 = ONNX 계약과 동일. 점유율/성공률 수렴 확인.

## Stage 5 — 배포 export & 라운드트립 (가장 중요)
- [ ] 학습된 **GAT .pt** 를 `Online-3D-BPP-PCT/export_onnx.py` 로 ONNX 변환.
- [ ] ONNX 입력 shape = **(1, 301, 9)**, 출력 = leaf 확률(100) 인지 확인.
- [ ] `submission_UOS-robostack/models/pct_model.onnx` 교체 후 `algorithm.py` 로 추론 라운드트립:
      박스 시퀀스 → packer.observe → ONNX argmax → packer.place → 위치. 점유율/성공 재현.
- [ ] **mass/density 정합 확인(주의 2)**: 학습이 setting 3(density 사용)이면 `algorithm.py`의
      `packer.observe(size)` 를 `observe(size, density=mass/vol/DENSITY_MAX)` 로 맞춰야 train/deploy 일치.
- [ ] Isaac 물리 채점 vs 베이스라인(92.3점) 비교로 회귀 없는지.

---

## 배포 호환성 메모 (확정 사항)
- ✅ action = leaf 인덱스, **위치는 packer가 디코딩** (NN은 인덱스만). 이식본도 동일 packer.py.
- ✅ **GAT 경로**는 순수 PCT 2709 관찰 사용 → ONNX 계약과 일치. 물리 피처는 reward/종료에만.
- ⚠️ **skrl/MLP 경로는 2714(물리 포함)** → algorithm.py 계약과 불일치. **export는 GAT로만.**
- ⚠️ `algorithm.py` 가 mass를 observe에 안 넘김(density=1.0). setting 3 학습과 어긋날 수 있음 → 대회 규정 확인.
