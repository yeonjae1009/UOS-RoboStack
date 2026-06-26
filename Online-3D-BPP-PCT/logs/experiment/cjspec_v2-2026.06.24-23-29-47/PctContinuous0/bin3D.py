from .space import Space
import numpy as np
import gym
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator
import torch
import random

class PackingContinuous(gym.Env):
    def __init__(self,
                 setting,
                 container_size=(10, 10, 10),
                 item_set=None, data_name=None, load_test_data=False,
                 internal_node_holder=80, leaf_node_holder=50, next_holder=1, shuffle=False,
                 sample_from_distribution = True,
                 sample_left_bound = 0.1,
                 sample_right_bound = 0.5,
                 **kwags):

        self.internal_node_holder = internal_node_holder
        self.leaf_node_holder = leaf_node_holder
        self.next_holder = next_holder

        self.shuffle = shuffle
        self.bin_size = container_size
        if sample_from_distribution:
            self.size_minimum = sample_left_bound
            self.sample_left_bound = sample_left_bound
            self.sample_right_bound = sample_right_bound
        else: self.size_minimum = np.min(np.array(item_set))
        self.setting = setting
        self.item_set = item_set
        # [CJ] setting 3: 박스 크기 → 실제 정규화 밀도 매핑 (givenData.item_density_set)
        self._den_map = {}
        # [CJ] 대회 랜덤 박스 스펙 (givenData 제공). 없으면 단일 bound 로 폴백.
        self.box_wl_bound = (sample_left_bound, sample_right_bound)
        self.box_h_bound = (sample_left_bound, sample_right_bound)
        self.box_mass_bound = None
        self.density_max = 1.0
        try:
            import givenData as _gd
            if hasattr(_gd, 'item_density_set'):
                for _s, _d in zip(_gd.item_size_set, _gd.item_density_set):
                    self._den_map[tuple(round(float(v), 6) for v in _s)] = float(_d)
            self.box_wl_bound = getattr(_gd, 'BOX_WL_BOUND', self.box_wl_bound)
            self.box_h_bound = getattr(_gd, 'BOX_H_BOUND', self.box_h_bound)
            self.box_mass_bound = getattr(_gd, 'BOX_MASS_BOUND', self.box_mass_bound)
            self.density_max = float(getattr(_gd, 'DENSITY_MAX', self.density_max))
        except Exception:
            self._den_map = {}
        # [CJ] 분포 샘플 시 가장 작은 박스 치수(=H 하한)를 size_minimum 으로 (EMS 필터 기준).
        if sample_from_distribution:
            self.size_minimum = min(self.box_wl_bound[0], self.box_h_bound[0])
        if self.setting == 2: self.orientation = 6
        else: self.orientation = 2

        # The class that maintains the contents of the bin.
        self.space = Space(*self.bin_size, self.size_minimum, self.internal_node_holder)

        # Generator for train/test data
        if not load_test_data:
            assert item_set is not None
            self.box_creator = RandomBoxCreator(item_set)
            assert isinstance(self.box_creator, BoxCreator)

        self.sample_from_distribution = sample_from_distribution
        if load_test_data:
            self.box_creator = LoadBoxCreator(data_name)

        self.test = load_test_data
        self.observation_space = gym.spaces.Box(low=0.0, high=self.space.height,
                                                shape=((self.internal_node_holder + self.leaf_node_holder + self.next_holder) * 9,))
        self.next_box_vec = np.zeros((self.next_holder, 9))

        self.LNES = 'EMS'  # Leaf Node Expansion Schemes: EMS

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            self.SEED = seed
        return [seed]

    # Calculate space utilization inside a bin.
    def get_box_ratio(self):
        coming_box = self.next_box
        return (coming_box[0] * coming_box[1] * coming_box[2]) / (self.space.plain_size[0] * self.space.plain_size[1] * self.space.plain_size[2])

    def _box_grid_slice(self, box, grid, nx, ny):
        ix0 = int(np.floor((box.lx + 1e-9) / grid))
        iy0 = int(np.floor((box.ly + 1e-9) / grid))
        ix1 = int(np.ceil((box.lx + box.x - 1e-9) / grid))
        iy1 = int(np.ceil((box.ly + box.y - 1e-9) / grid))

        ix0 = max(0, min(nx, ix0))
        iy0 = max(0, min(ny, iy0))
        ix1 = max(ix0, min(nx, ix1))
        iy1 = max(iy0, min(ny, iy1))

        return ix0, ix1, iy0, iy1

    def _build_height_map(self, boxes, grid=0.025):
        pallet_x, pallet_y, _ = self.space.plain_size
        nx = max(1, int(np.ceil(pallet_x / grid)))
        ny = max(1, int(np.ceil(pallet_y / grid)))
        height_map = np.zeros((nx, ny), dtype=np.float32)

        for box in boxes:
            ix0, ix1, iy0, iy1 = self._box_grid_slice(box, grid, nx, ny)
            if ix1 <= ix0 or iy1 <= iy0:
                continue

            top_height = float(box.lz + box.z)
            height_map[ix0:ix1, iy0:iy1] = np.maximum(
                height_map[ix0:ix1, iy0:iy1],
                top_height,
            )

        return height_map

    def _layout_metrics(self, grid=0.025):
        pallet_x, pallet_y, pallet_z = self.space.plain_size
        nx = max(1, int(np.ceil(pallet_x / grid)))
        ny = max(1, int(np.ceil(pallet_y / grid)))

        floor_map = np.zeros((nx, ny), dtype=bool)
        for box in self.space.boxes:
            if abs(box.lz) > 1e-6:
                continue

            ix0, ix1, iy0, iy1 = self._box_grid_slice(box, grid, nx, ny)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            floor_map[ix0:ix1, iy0:iy1] = True

        height_map = self._build_height_map(self.space.boxes, grid)
        occupied = height_map > 1e-6
        if np.any(occupied):
            height_roughness = float(np.std(height_map[occupied]) / max(float(pallet_z), 1e-9))
        else:
            height_roughness = 0.0

        x_centers = (np.arange(nx) + 0.5) * grid
        y_centers = (np.arange(ny) + 0.5) * grid
        xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')

        band = 0.13
        boundary_mask = (
            (xx <= band) |
            (xx >= float(pallet_x) - band) |
            (yy <= band) |
            (yy >= float(pallet_y) - band)
        )

        corner = 0.22
        corner_mask = (
            ((xx <= corner) | (xx >= float(pallet_x) - corner)) &
            ((yy <= corner) | (yy >= float(pallet_y) - corner))
        )

        floor_coverage = float(np.mean(floor_map))
        boundary_floor_coverage = (
            float(np.sum(floor_map & boundary_mask) / max(np.sum(boundary_mask), 1))
        )
        corner_floor_coverage = (
            float(np.sum(floor_map & corner_mask) / max(np.sum(corner_mask), 1))
        )

        return {
            'floor_coverage': floor_coverage,
            'boundary_floor_coverage': boundary_floor_coverage,
            'corner_floor_coverage': corner_floor_coverage,
            'height_roughness': height_roughness,
        }

    def _support_ratio(self, packed_box):
        if abs(packed_box.lz) <= 1e-6:
            return 1.0

        support_area = 0.0
        for edge in packed_box.bottom_edges:
            if edge.area is None:
                continue
            x1, y1, x2, y2 = edge.area
            support_area += max(0.0, x2 - x1) * max(0.0, y2 - y1)

        base_area = max(float(packed_box.x * packed_box.y), 1e-9)
        return float(np.clip(support_area / base_area, 0.0, 1.0))

    def _compute_shaped_reward(self, box_ratio, packed_box, before, after):
        volume_reward = box_ratio * 10.0

        delta_floor = after['floor_coverage'] - before['floor_coverage']
        delta_boundary = after['boundary_floor_coverage'] - before['boundary_floor_coverage']
        delta_corner = after['corner_floor_coverage'] - before['corner_floor_coverage']

        floor_coverage_reward = 1.0 * delta_floor
        boundary_floor_reward = 0.8 * delta_boundary
        corner_floor_reward = 0.6 * delta_corner

        height_delta = before['height_roughness'] - after['height_roughness']
        height_smoothness_reward = 0.5 * float(np.clip(height_delta, -0.05, 0.05))

        support_ratio = self._support_ratio(packed_box)
        support_reward = 0.05 * support_ratio
        if packed_box.lz > 1e-6:
            weak_support_penalty = 0.05 * max(0.0, 0.85 - support_ratio)
        else:
            weak_support_penalty = 0.0

        reward = (
            volume_reward
            + floor_coverage_reward
            + boundary_floor_reward
            + corner_floor_reward
            + height_smoothness_reward
            + support_reward
            - weak_support_penalty
        )

        self.last_reward_terms = {
            'volume_reward': float(volume_reward),
            'floor_coverage_reward': float(floor_coverage_reward),
            'boundary_floor_reward': float(boundary_floor_reward),
            'corner_floor_reward': float(corner_floor_reward),
            'height_smoothness_reward': float(height_smoothness_reward),
            'support_reward': float(support_reward),
            'weak_support_penalty': float(weak_support_penalty),
            'support_ratio': float(support_ratio),
            'reward': float(reward),
        }

        return float(reward)

    def reset(self):
        self.box_creator.reset()
        self.packed = []
        self.space.reset()
        self.box_creator.generate_box_size()
        cur_observation = self.cur_observation()
        return cur_observation

    # Count and return all PCT nodes.
    def cur_observation(self):
        boxes = []
        leaf_nodes = []
        self.next_box = self.gen_next_box()
        if self.test:
            if self.setting == 3: self.next_den = self.next_box[3]
            else: self.next_den = 1
            self.next_box = [round(self.next_box[0], 3), round(self.next_box[1], 3), round(self.next_box[2], 3)]
        else:
            if self.setting < 3: self.next_den = 1
            elif self.sample_from_distribution:
                if self.box_mass_bound is not None:
                    # [CJ] 대회 스펙: 질량 ~ U(0.5,6.0)kg 독립 샘플 → 밀도=질량/부피 → DENSITY_MAX 로 정규화([0,1]).
                    vol = max(self.next_box[0] * self.next_box[1] * self.next_box[2], 1e-9)
                    mass = np.random.uniform(self.box_mass_bound[0], self.box_mass_bound[1])
                    self.next_den = float((mass / vol) / self.density_max)
                else:
                    # 폴백: 밀도 U(0,1)
                    self.next_den = np.random.random()
            else:
                # [CJ] 고정 5종 모드: 크기 → 실제 정규화 밀도 매핑 (없으면 랜덤)
                _key = tuple(round(float(v), 6) for v in self.next_box)
                self.next_den = self._den_map.get(_key, np.random.random())

        boxes.append(self.space.box_vec)
        leaf_nodes.append(self.get_possible_position())

        next_box = sorted(list(self.next_box))
        self.next_box_vec[:, 3:6] = next_box
        self.next_box_vec[:, 0] = self.next_den
        self.next_box_vec[:, -1] = 1
        return np.reshape(np.concatenate((*boxes, *leaf_nodes, self.next_box_vec)), (-1))

    # Generate the next item to be placed.
    def gen_next_box(self):
        if self.sample_from_distribution and not self.test:
            # [CJ] 대회 스펙 랜덤: W,L ~ U(0.17,0.32), H(z, 항상 수직) ~ U(0.13,0.26).
            #      orientation=2 라 z축이 높이로 고정되고 x↔y만 회전(대회 Z 0/90 일치).
            wl, h = self.box_wl_bound, self.box_h_bound
            next_box = (round(np.random.uniform(wl[0], wl[1]), 3),   # W (x)
                        round(np.random.uniform(wl[0], wl[1]), 3),   # L (y)
                        round(np.random.uniform(h[0],  h[1]),  3))   # H (z)
        else:
            next_box = self.box_creator.preview(1)[0]
        return next_box

    # Detect potential leaf nodes and check their feasibility.
    def get_possible_position(self):
        if   self.LNES == 'EMS':
            allPostion = self.space.EMSPoint(self.next_box, self.setting)
        elif self.LNES == 'EV':
            allPostion = self.space.EventPoint(self.next_box, self.setting)
        else:
            assert False, 'Wrong LNES'

        if self.shuffle:
            np.random.shuffle(allPostion)

        leaf_node_idx = 0
        leaf_node_vec = np.zeros((self.leaf_node_holder, 9))
        tmp_list = []

        for position in allPostion:
            xs, ys, zs, xe, ye, ze = position
            x = xe - xs
            y = ye - ys
            z = ze - zs

            if self.space.drop_box_virtual([x, y, z], (xs, ys), False, self.next_den, self.setting):
                tmp_list.append([xs, ys, zs, xe, ye, self.bin_size[2], 0, 0, 1])
                leaf_node_idx += 1

            if leaf_node_idx >= self.leaf_node_holder: break

        if len(tmp_list) != 0:
            leaf_node_vec[0:len(tmp_list)] = np.array(tmp_list)

        return leaf_node_vec

    # Convert the selected leaf node to the placement of the current item.
    def LeafNode2Action(self, leaf_node):
        if np.sum(leaf_node[0:6]) == 0: return (0, 0, 0), self.next_box
        x = round(leaf_node[3] - leaf_node[0], 6)
        y = round(leaf_node[4] - leaf_node[1], 6)
        record = [0,1,2]
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

    def step(self, action):
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)

        if len(action_array) != 3 and np.sum(action_array[0:6]) == 0:
            leaf_nodes = self.get_possible_position()
            no_feasible_leaf = float(leaf_nodes[:, 8].sum()) <= 0.0
            reward = 0.0 if no_feasible_leaf else -10.0
            done = True
            ratio = self.space.get_ratio()
            termination = 'no_feasible_leaf' if no_feasible_leaf else 'hard_fail'
            info = {
                'counter': len(self.space.boxes),
                'ratio': ratio,
                'reward': ratio * 10 if no_feasible_leaf else 0.0,
                'termination': termination,
            }
            return self.cur_observation(), reward, done, info

        if len(action_array) != 3:
            action, next_box = self.LeafNode2Action(action_array)
        else:
            action = action_array
            next_box = self.next_box

        idx = [round(action[1], 6), round(action[2], 6)]
        bin_index = 0
        rotation_flag = action[0]

        before_metrics = self._layout_metrics()
        box_ratio = self.get_box_ratio()
        succeeded = self.space.drop_box(next_box, idx, rotation_flag, self.next_den, self.setting)

        if not succeeded:
            reward = -10.0
            done = True
            info = {
                'counter': len(self.space.boxes),
                'ratio': self.space.get_ratio(),
                'reward': 0.0,
                'termination': 'hard_fail',
            }
            return self.cur_observation(), reward, done, info

        ################################################
        ############# cal leaf nodes here ##############
        ################################################
        packed_box = self.space.boxes[-1]
        after_metrics = self._layout_metrics()

        if  self.LNES == 'EMS':
            self.space.GENEMS([packed_box.lx, packed_box.ly, packed_box.lz,
                                       round(packed_box.lx + packed_box.x, 6),
                                       round(packed_box.ly + packed_box.y, 6),
                                       round(packed_box.lz + packed_box.z, 6)])

        self.packed.append(
            [packed_box.x, packed_box.y, packed_box.z, packed_box.lx, packed_box.ly, packed_box.lz, bin_index])

        self.box_creator.drop_box()  # remove current box from the list
        self.box_creator.generate_box_size()  # add a new box to the list
        reward = self._compute_shaped_reward(box_ratio, packed_box, before_metrics, after_metrics)

        done = False
        info = dict()
        info['counter'] = len(self.space.boxes)
        info['reward_terms'] = self.last_reward_terms
        return self.cur_observation(), reward, done, info
