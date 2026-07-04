# container_size: A vector of length 3 describing the size of the container in the x, y, z dimension.
# item_size_set:  A list records the size of each item. The size of each item is also described by a vector of length 3.

# === CJ 팔레타이징 대회 스펙 ===
# 팔레트: 1.2(L) x 1.0(W) x 1.25(H) m
# 박스 5종 (대회 box_sequence에서 추출, 단위 m)
container_size = [1.2, 1.0, 1.25]

item_size_set = [
    (0.195, 0.178, 0.134),   # type 0
    (0.245, 0.178, 0.14),    # type 1
    (0.245, 0.22,  0.158),   # type 2
    (0.31,  0.233, 0.21),    # type 3
    (0.315, 0.272, 0.257),   # type 4
]

# [CJ] setting 3 학습용 실제 밀도(정규화). 실제 밀도 = mass/volume 을 최대밀도(272.5)로 나눈 값.
# (COM 안정성은 스케일 무관이라 정규화해도 정확. 추론 때도 동일 스케일 272.5 로 정규화해 입력)
# 순서는 item_size_set 와 동일.
DENSITY_SCALE = 272.5
item_density_set = [
    0.395,   # type 0  (107.5 / 272.5)
    0.601,   # type 1  (163.8 / 272.5)
    0.862,   # type 2  (234.8 / 272.5)
    0.968,   # type 3  (263.7 / 272.5)
    1.000,   # type 4  (272.5 / 272.5)
]

# 학습 시 RandomBoxCreator가 위 5종에서 무작위로 박스를 뽑아 시퀀스를 실시간 생성한다.
# (--sample-from-distribution 을 쓰지 않아야 이 item_size_set 에서 샘플링됨)
# setting 3 일 때 bin3D.py 가 크기→item_density_set 밀도를 매핑해 next_den 으로 사용.

# === CJ 대회 "랜덤 박스" 스펙 (대회측 제시 범위) ===
# W(170~320), L(170~320), H(130~260) mm, 질량 0.5~6.0 kg. 단위 m / kg.
# --sample-from-distribution 학습 시 bin3D.gen_next_box / cur_observation 이 이 범위로 생성한다.
# 회전은 orientation=2(설정 1·3) → z축(높이)이 항상 수직, x↔y만 스왑(대회 Z 0/90 일치).
# 따라서 박스 3축 중 (x,y)=W,L 범위, (z)=H 범위로 둔다.
BOX_WL_BOUND   = (0.17, 0.32)   # W(x), L(y) [m]
BOX_H_BOUND    = (0.13, 0.26)   # H(z, 높이) [m]
BOX_MASS_BOUND = (0.5, 6.0)     # 질량 [kg]
# 밀도(=질량/부피) 정규화 상수: 네트워크 입력을 [0,1]로 유지하려면 가능한 최대밀도로 나눈다.
#   최대밀도 = 최대질량 / 최소부피 = 6.0 / (0.17·0.17·0.13).  COM 안정성은 스케일 무관이라
#   정규화 상수는 물리에 영향 없음(상대질량만 중요). 추론 코드도 동일 상수로 정규화해야 함.
DENSITY_MAX    = BOX_MASS_BOUND[1] / (BOX_WL_BOUND[0] * BOX_WL_BOUND[0] * BOX_H_BOUND[0])
