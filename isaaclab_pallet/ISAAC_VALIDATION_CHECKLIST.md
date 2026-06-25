# Isaac 머신 검증 체크리스트 (isaaclab_pallet)

이 머신(Isaac 없음)에서 이미 **수치로 통과한 것**과, **Isaac 머신에서만 검증 가능한 것**을 구분한다.
목표 배포 형태: **학습된 GAT(.pt) → ONNX → `submission_UOS-robostack/algorithm.py` 추론**
(입력 = packer가 만든 PCT 그래프 관찰, 출력 = leaf 인덱스, 위치는 packer가 디코딩).

---

## 이미 통과 (이 머신, Isaac 불필요) — 재현용
- [ ] CPU layer 동일성: `python3 isaaclab_pallet/scripts/test_equivalence.py --max-boxes 250 --policy cycle`
      → 원본 PCT vs 이식 reward/metric **0.000e+00** (bit-exact) 확인.
- [ ] 병렬 packer 정확성/속도: `python3 isaaclab_pallet/scripts/test_packer_pool.py --num-envs 16 --workers 4`
      → 직렬 vs 병렬 **0 불일치**, 속도 향상(약 3x) 확인.
- [ ] 박스 연속 생성: 생성기가 W,L∈[0.17,0.32], H∈[0.13,0.26], 질량∈[0.5,6.0] **연속**(고정 5종 아님) 확인.

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

## Stage 1.5 — 박스 분포 & 순서 다양성 (규정 정합)
- [ ] `random_boxes=True`(기본)로 학습 박스가 **규정 연속 랜덤**인지(고정 5종 JSON 아님).
- [ ] `shuffle_each_episode=True`(기본): reset마다 공급 **순서**가 바뀌는지(env·에피소드별 상이, 시드 재현).
      → 규정 "박스 랜덤 순서 제공"과 부합. 크기 자체의 에피소드별 재샘플은 prim rescale 필요(아래 보류 항목).
- [ ] 크기 다양성 더 필요하면 `max_boxes`↑(풀 샘플 확대) 또는 `box_seed` 바꿔 재시작.

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

## Stage 6 — 제출 제약 (확인 완료 + 마무리)
- [x] **패키지**: 추론은 `requirements.txt`의 numpy+onnxruntime+pyyaml(+matplotlib)만. torch/skrl은 학습용 → 제출 미포함. GAT→ONNX면 충족.
- [ ] **numpy 2.x 정합**: 제출 환경 `numpy==2.4.3`. 원본 `convex_hull.py`의 `np.float`(1.24 제거)는 2.x에서 터짐
      → 제출엔 **`np.float64`로 고친 `templete code` 버전** packer/convex_hull 포함 확인.
- [ ] **시간 ≤ 90초/데이터셋**: 250박스 추론(observe=EMS+convex-hull 매 박스)이 90초 안인지 실측. 병목이면 packer 최적화/병렬.
- [ ] **구조**: ZIP 최상위 `main.py` + `algorithm.py` + `config/*.yaml`. (`submission_UOS-robostack`가 이미 이 구조)
- [ ] **출력 포맷**: Box ID, position=centroid(x,y,z), rotation∈{0,90}, 팔레트 좌하단=(0,0,0). 제공 코드가 생성하므로 임의 변경 금지.

---

## 규정 정합 (확정 사항)
- ✅ **팔레트 축**: 제공 `algorithm_config.yaml` `length 1.2 / width 1.0 / height 1.25` = `container=[1.2,1.0,1.25]`.
  이식본 `pallet_size=(1.2,1.0,1.25)`와 일치. (규정 본문 "가로/세로"보다 **제공 템플릿이 권위 기준**.)
  확인법: 출력 JSON을 `palletizing_simulator`에 넣어 OOB·낙하 없이 안착하는지.
- ✅ **박스 스펙**: W,L 0.17–0.32 / H 0.13–0.26 m / 질량 0.5–6.0 kg **연속 랜덤** 생성. (고정 5종 JSON은 테스트용만)
- ✅ **랜덤 순서**: 에피소드마다 공급 순서 셔플.
- ❌ **버퍼/박스 선택**: 미구현(사용자 보류). 현재 순수 online. 규정의 버퍼 가산점(최대 +20)·선택 전략은 별도 설계 필요.

## 배포 호환성 메모 (확정 사항)
- ✅ action = leaf 인덱스, **위치는 packer가 디코딩** (NN은 인덱스만). 이식본도 동일 packer.py.
- ✅ **GAT 경로**는 순수 PCT 2709 관찰 사용 → ONNX 계약과 일치. 물리 피처는 reward/종료에만.
- ⚠️ **skrl/MLP 경로는 2714(물리 포함)** → algorithm.py 계약과 불일치. **export는 GAT로만.**
- ⚠️ `algorithm.py` 가 mass를 observe에 안 넘김(density=1.0). setting 3 학습과 어긋날 수 있음 → 대회 규정 확인.

## 보류 (Isaac 머신 / 추후)
- 에피소드별 **박스 크기 재샘플**(prim rescale) — 현재는 순서만 셔플, 크기 풀은 고정.
- **버퍼/박스 선택** 전략 (점수 핵심, 큰 작업).
