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

# 학습 시 RandomBoxCreator가 위 5종에서 무작위로 박스를 뽑아 시퀀스를 실시간 생성한다.
# (--sample-from-distribution 을 쓰지 않아야 이 item_size_set 에서 샘플링됨)
