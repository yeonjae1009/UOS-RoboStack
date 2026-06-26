"""
PCT 모델 → 대회(algorithm_results) JSON 변환기.

학습된 PCT 체크포인트로 '대회 box_sequence(JSONL)'를 온라인(버퍼0) 적재한 뒤,
그 결과를 대회 시뮬레이터가 읽는 algorithm_results JSON 형식으로 저장한다.
=> 저장된 JSON을 palletizing_simulator/algorithm_results/ 로 복사하면 Isaac Sim(run_gui.sh)으로 볼 수 있다.

사용:
  .venv-pct/bin/python export_to_contest.py \
      --model-path logs/experiment/<run>/PCT-....pt \
      --box-sequence "../templete code/box_sequence/box_sequence_0.json" \
      --out out_box_sequence_0.json \
      --internal-node-holder 200 --leaf-node-holder 100

주의: holder 값은 반드시 학습 때와 동일해야 한다(네트워크 입력 크기 일치).
"""
import sys, json, argparse
import numpy as np
import torch

# 1) 내 인자 먼저 파싱
mine = argparse.ArgumentParser()
mine.add_argument('--model-path', required=True)
mine.add_argument('--box-sequence', required=True)
mine.add_argument('--out', required=True)
mine.add_argument('--internal-node-holder', default='200')
mine.add_argument('--leaf-node-holder', default='100')
mine.add_argument('--setting', default='1')
margs, _ = mine.parse_known_args()

# 2) PCT 표준 args 구성 (get_args 가 sys.argv 를 읽으므로 갈아끼움)
sys.argv = ['export', '--continuous', '--setting', str(margs.setting),
            '--internal-node-holder', str(margs.internal_node_holder),
            '--leaf-node-holder', str(margs.leaf_node_holder),
            '--evaluate', '--load-model', '--model-path', margs.model_path]

from tools import registration_envs, get_args, get_leaf_nodes_with_factor, load_policy
from model import DRL_GAT
from pct_envs.PctContinuous0.binCreator import BoxCreator
import gym

registration_envs()
args = get_args()
device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')


# 3) 대회 박스 시퀀스 로드 (JSONL: step/id/size/mass)
def load_seq(path):
    boxes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                boxes.append(json.loads(line))
    return boxes

seq = load_seq(margs.box_sequence)
sizes = [list(map(float, b['size'])) for b in seq]


# 4) 대회 시퀀스를 그대로 공급하는 박스 생성기 (온라인, 순서대로)
class FixedCreator(BoxCreator):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
        self.idx = 0
    def reset(self):
        self.box_list.clear()
        self.idx = 0
    def generate_box_size(self, **kw):
        if self.idx < len(self.sizes):
            self.box_list.append(tuple(self.sizes[self.idx]))
            self.idx += 1
        else:
            self.box_list.append((100.0, 100.0, 100.0))  # 못 놓는 sentinel → 에피소드 종료


env = gym.make(args.id, setting=args.setting,
               container_size=args.container_size, item_set=args.item_size_set,
               internal_node_holder=args.internal_node_holder,
               leaf_node_holder=args.leaf_node_holder,
               LNES=args.lnes, shuffle=False,
               sample_from_distribution=False,
               data_name=None, load_test_data=False)
e = env.unwrapped
e.box_creator = FixedCreator(sizes)

# [CJ] setting 3 density 정렬: 대회 박스 mass 로부터 학습과 동일하게 정규화(=mass/부피/DENSITY_MAX).
#      export 는 sample_from_distribution=False 라 env 가 _den_map[사이즈] 조회를 쓰므로,
#      그 맵을 대회 시퀀스(사이즈→밀도)로 채워야 학습/추론 density 입력이 일치한다.
try:
    import givenData as _gd
    _DMAX = float(getattr(_gd, 'DENSITY_MAX', 1.0))
except Exception:
    _DMAX = 1.0
e._den_map = {}
for _b in seq:
    _sz = tuple(round(float(v), 6) for v in _b['size'])
    _vol = float(_b['size'][0]) * float(_b['size'][1]) * float(_b['size'][2])
    e._den_map[_sz] = (float(_b['mass']) / max(_vol, 1e-9)) / _DMAX
print(f"[density] DENSITY_MAX={_DMAX:.1f}, mapped {len(e._den_map)} sizes (setting={args.setting})")

policy = DRL_GAT(args).to(device)
policy = load_policy(margs.model_path, policy)
policy.eval()

factor = args.normFactor
batchX = torch.arange(1)


def to_nodes(obs):
    t = torch.FloatTensor(obs).to(device).unsqueeze(0)
    return get_leaf_nodes_with_factor(t, 1, args.internal_node_holder, args.leaf_node_holder)


obs = env.reset()
all_nodes, leaf_nodes = to_nodes(obs)
steps = 0
while True:
    with torch.no_grad():
        _, selectedIdx, _, _ = policy(all_nodes, True, normFactor=factor)
    selected_leaf_node = leaf_nodes[batchX, selectedIdx.squeeze()]
    obs, reward, done, info = env.step(selected_leaf_node.cpu().numpy()[0][0:6])
    steps += 1
    if done:
        break
    all_nodes, leaf_nodes = to_nodes(obs)

placements = e.packed            # [[x,y,z, lx,ly,lz, bin], ...] 적재 순서
ratio = float(e.space.get_ratio())
n_placed = len(placements)
print(f"placed {n_placed}/{len(seq)} boxes,  space ratio = {ratio:.4f}")

# 5) 대회 JSON 형식으로 변환
out_sequence = []
for i, p in enumerate(placements):
    x, y, z, lx, ly, lz = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])
    b = seq[i]
    L, W, H = float(b['size'][0]), float(b['size'][1]), float(b['size'][2])
    # 회전 판정: 배치된 (x,y)가 원본 (L,W)면 0°, (W,L)면 90°
    rot = 0 if (abs(x - L) < 1e-3 and abs(y - W) < 1e-3) else 90
    out_sequence.append({
        "step": int(b['step']),
        "id": int(b['id']),
        "size": [round(x, 3), round(y, 3), round(z, 3)],
        "mass": float(b['mass']),
        "position": [round(lx + x / 2.0, 3), round(ly + y / 2.0, 3), round(lz + z / 2.0, 3)],
        "rotation": int(rot),
    })

terminated = n_placed < len(seq)
result = {
    "buffer_size": 0,
    "sequence": out_sequence,
    "terminated": terminated,
    "terminated_step": int(seq[n_placed]['step']) if terminated else None,
    "finished_by_user": False,
}
with open(margs.out, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"saved -> {margs.out}")
