"""
PCT 온라인 패킹 드라이버 (torch/gym 없이 순수 numpy).

원본 pct_envs/PctContinuous0/bin3D.py 의 PackingContinuous 에서
관찰값 생성(cur_observation/get_possible_position)과 배치(step) 로직만 추출하고,
torch.seed / gym / box_creator 결합을 제거했다. 박스는 외부에서 한 개씩 주입한다.

흐름:
  reset()                  -> 빈 팔레트로 초기화
  observe(next_box)        -> 현재 박스에 대한 관찰값(flat) 반환  (신경망 입력)
  place(leaf_node[0:6])    -> 선택된 잎 노드 위치에 배치, 성공 여부 반환
  packed                   -> [[x,y,z, lx,ly,lz, bin], ...] 배치 기록(적재 순서)
"""
import numpy as np
from .space import Space


class Packer:
    def __init__(self, container_size, size_minimum,
                 internal_node_holder=200, leaf_node_holder=100, setting=1):
        self.bin_size = list(container_size)
        self.internal_node_holder = internal_node_holder
        self.leaf_node_holder = leaf_node_holder
        self.next_holder = 1
        self.setting = setting
        self.size_minimum = size_minimum
        self.space = Space(*self.bin_size, size_minimum, internal_node_holder)
        self.next_box_vec = np.zeros((self.next_holder, 9))
        self.next_box = None
        self.next_den = 1
        self.packed = []

    def reset(self):
        self.space.reset()
        self.packed = []

    # 현재 박스에 대한 잎 노드(배치 후보) 생성
    def get_possible_position(self):
        allPosition = self.space.EMSPoint(self.next_box, self.setting)
        leaf_node_idx = 0
        leaf_node_vec = np.zeros((self.leaf_node_holder, 9))
        tmp_list = []
        for position in allPosition:
            xs, ys, zs, xe, ye, ze = position
            x = xe - xs
            y = ye - ys
            z = ze - zs
            if self.space.drop_box_virtual([x, y, z], (xs, ys), False, self.next_den, self.setting):
                tmp_list.append([xs, ys, zs, xe, ye, self.bin_size[2], 0, 0, 1])
                leaf_node_idx += 1
            if leaf_node_idx >= self.leaf_node_holder:
                break
        if len(tmp_list) != 0:
            leaf_node_vec[0:len(tmp_list)] = np.array(tmp_list)
        return leaf_node_vec

    # 관찰값(flat vector) 생성: [내부노드 | 잎노드 | 다음박스]
    #   density: setting 3 학습 시 next_box_vec[:,0] 에 들어간 정규화 밀도(=mass/부피/DENSITY_MAX).
    #            잎 노드 생성(drop_box_virtual)이 next_den 으로 안정성을 보므로 관찰 전에 설정해야 함.
    def observe(self, next_box, density=1.0):
        self.next_box = [float(next_box[0]), float(next_box[1]), float(next_box[2])]
        self.next_den = float(density)
        boxes = [self.space.box_vec]
        leaf_nodes = [self.get_possible_position()]
        nb_sorted = sorted(list(self.next_box))
        self.next_box_vec[:, 3:6] = nb_sorted
        self.next_box_vec[:, 0] = self.next_den
        self.next_box_vec[:, -1] = 1
        return np.reshape(np.concatenate((*boxes, *leaf_nodes, self.next_box_vec)), (-1))

    # 잎 노드(앞 6개 값: xs,ys,zs,xe,ye,ze)를 실제 박스 배치 동작으로 변환
    def _leaf_to_action(self, leaf_node):
        if np.sum(leaf_node[0:6]) == 0:
            return (0, 0, 0), tuple(self.next_box)
        x = round(leaf_node[3] - leaf_node[0], 6)
        y = round(leaf_node[4] - leaf_node[1], 6)
        record = [0, 1, 2]
        for r in record:
            if abs(x - self.next_box[r]) < 1e-6:
                record.remove(r)
                break
        for r in record:
            if abs(y - self.next_box[r]) < 1e-6:
                record.remove(r)
                break
        z = self.next_box[record[0]]
        action = (0, leaf_node[0], leaf_node[1])
        next_box = (x, y, z)
        return action, next_box

    # 선택된 잎 노드에 박스 배치. 성공하면 True 와 packed 기록 추가.
    def place(self, leaf_node):
        action, next_box = self._leaf_to_action(leaf_node)
        idx = [round(action[1], 6), round(action[2], 6)]
        rotation_flag = action[0]
        ok = self.space.drop_box(next_box, idx, rotation_flag, self.next_den, self.setting)
        if not ok:
            return False
        pb = self.space.boxes[-1]
        self.space.GENEMS([pb.lx, pb.ly, pb.lz,
                           round(pb.lx + pb.x, 6),
                           round(pb.ly + pb.y, 6),
                           round(pb.lz + pb.z, 6)])
        self.packed.append([pb.x, pb.y, pb.z, pb.lx, pb.ly, pb.lz, 0])
        return True

    def get_ratio(self):
        return self.space.get_ratio()
