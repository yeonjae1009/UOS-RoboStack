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
